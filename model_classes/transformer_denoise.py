



from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce
from einops_exts import rearrange_many, repeat_many, check_shape
import torch.nn.functional as F
from einops_exts.torch import EinopsToAndFrom



import math
from functools import partial, wraps
from torch import nn, einsum
from pathlib import Path


from utils import default, once,  exists, cast_tuple,resize_image_to
print_once = once(print)

import  torch

def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)

# tensor helpers


def l2norm(t):
    return F.normalize(t, dim=-1)


# classifier free guidance functions

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob





def masked_mean(t, *, dim, mask=None):
    if not exists(mask):
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, 'b n -> b n 1')
    masked_t = t.masked_fill(~mask, 0.)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)





class Always():
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Parallel(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        outputs = [fn(x) for fn in self.fns]
        return sum(outputs)











# norms and residuals

class LayerNorm(nn.Module):
    def __init__(self, feats, stable=False, dim=-1):
        super().__init__()
        self.stable = stable
        self.dim = dim

        self.g = nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim

        if self.stable:
            x = x / x.amax(dim=dim, keepdim=True).detach()

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=dim, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=dim, keepdim=True)

        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)


# attention pooling

class PerceiverAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_head=64,
            heads=8,
            cosine_sim_attn=False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5 if not cosine_sim_attn else 1
        self.cosine_sim_attn = cosine_sim_attn
        self.cosine_sim_scale = 16 if cosine_sim_attn else 1

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.LayerNorm(dim)
        )

    def forward(self, x, latents, mask=None):
        x = self.norm(x)
        latents = self.norm_latents(latents)

        b, h = x.shape[0], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values
        # derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=h)

        q = q * self.scale

        # cosine sim attention

        if self.cosine_sim_attn:
            q, k = map(l2norm, (q, k))

        # similarities and masking

        sim = einsum('... i d, ... j d  -> ... i j', q, k) * self.cosine_sim_scale

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value=True)

            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.to(sim.dtype)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)

# helper classes

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class PerceiverResampler(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            num_latents=64,
            num_latents_mean_pooled=4,  # number of latents derived from mean pooled representation of the sequence
            max_seq_len=512,
            ff_mult=4,
            cosine_sim_attn=False
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.to_latents_from_mean_pooled_seq = None

        if num_latents_mean_pooled > 0:
            self.to_latents_from_mean_pooled_seq = nn.Sequential(
                LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange('b (n d) -> b n d', n=num_latents_mean_pooled)
            )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads, cosine_sim_attn=cosine_sim_attn),
                FeedForward(dim=dim, mult=ff_mult)
            ]))

    def forward(self, x, mask=None):
        n, device = x.shape[1], x.device
        pos_emb = self.pos_emb(torch.arange(n, device=device))

        x_with_pos = x + pos_emb

        latents = repeat(self.latents, 'n d -> b n d', b=x.shape[0])

        if exists(self.to_latents_from_mean_pooled_seq):
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x_with_pos, latents, mask=mask) + latents
            latents = ff(latents) + latents

        return latents


# attention

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            *,
            dim_head=64,
            heads=8,
            context_dim=None,
            cosine_sim_attn=False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5 if not cosine_sim_attn else 1.
        self.cosine_sim_attn = cosine_sim_attn
        self.cosine_sim_scale = 16 if cosine_sim_attn else 1

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, dim_head * 2)) if exists(
            context_dim) else None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            LayerNorm(dim)
        )

    def forward(self, x, context=None, mask=None, attn_bias=None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        q = q * self.scale

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim=-2), 'd -> b 1 d', b=b)
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # add text conditioning, if present

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim=-1)
            k = torch.cat((ck, k), dim=-2)
            v = torch.cat((cv, v), dim=-2)

        # cosine sim attention

        if self.cosine_sim_attn:
            q, k = map(l2norm, (q, k))

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.cosine_sim_scale

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)

            mask = rearrange(mask, 'b j -> b 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.to(sim.dtype)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# decoder

def Upsample(dim, dim_out=None):
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv1d(dim, dim_out, 3, padding=1)
    )


class PixelShuffleUpsample(nn.Module):
    """
    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """

    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv1d(dim, dim_out * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


def Downsample(dim, dim_out=None):
    # https://arxiv.org/abs/2208.03641 shows this is the most optimal way to downsample
    # named SP-conv in the paper, but basically a pixel unshuffle

    dim_out = default(dim_out, dim)

    return nn.Sequential(

        Rearrange('b c (h s1)  -> b (c s1) h', s1=2),
        nn.Conv1d(dim * 2, dim_out, 1)

    )


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class Block(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            groups=8,
            norm=True
    ):
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else Identity()
        self.activation = nn.SiLU()
        self.project = nn.Conv1d(dim, dim_out, 3, padding=1)

    def forward(self, x, scale_shift=None):
        x = self.groupnorm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return self.project(x)


class ResnetBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            *,
            cond_dim=None,
            time_cond_dim=None,
            groups=8,
            linear_attn=False,
            use_gca=False,
            squeeze_excite=False,
            **attn_kwargs
    ):
        super().__init__()

        self.time_mlp = None

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2)
            )

        self.cross_attn = None

        if exists(cond_dim):
            attn_klass = CrossAttention if not linear_attn else LinearCrossAttention

            self.cross_attn = EinopsToAndFrom(

                'b c h ',
                'b h c',
                attn_klass(
                    dim=dim_out,
                    context_dim=cond_dim,
                    **attn_kwargs
                )
            )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        self.gca = GlobalContext(dim_in=dim_out, dim_out=dim_out) if use_gca else Always(1)

        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else Identity()

    def forward(self, x, time_emb=None, cond=None):

        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)

            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x)

        if exists(self.cross_attn):
            assert exists(cond)
            h = self.cross_attn(h, context=cond) + h

        h = self.block2(h, scale_shift=scale_shift)

        h = h * self.gca(h)

        return h + self.res_conv(x)


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            *,
            context_dim=None,
            dim_head=64,
            heads=8,
            norm_context=False,
            cosine_sim_attn=False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5 if not cosine_sim_attn else 1.
        self.cosine_sim_attn = cosine_sim_attn
        self.cosine_sim_scale = 16 if cosine_sim_attn else 1

        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else Identity()

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            LayerNorm(dim)
        )

    def forward(self, x, context, mask=None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim=-2), 'd -> b h 1 d', h=self.heads, b=b)

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        q = q * self.scale

        # cosine sim attention

        if self.cosine_sim_attn:
            q, k = map(l2norm, (q, k))

        # similarities

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.cosine_sim_scale

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)

            mask = rearrange(mask, 'b j -> b 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.to(sim.dtype)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class LinearCrossAttention(CrossAttention):
    def forward(self, x, context, mask=None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> (b h) n d', h=self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim=-2), 'd -> (b h) 1 d', h=self.heads, b=b)

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # masking

        max_neg_value = -torch.finfo(x.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b n -> b n 1')
            k = k.masked_fill(~mask, max_neg_value)
            v = v.masked_fill(~mask, 0.)

        # linear attention

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=32,
            heads=8,
            dropout=0.05,
            context_dim=None,
            **kwargs
    ):
        super().__init__()
        ChanLayerNorm = partial(LayerNorm, dim=-2)
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)

        self.nonlin = nn.SiLU()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(dim, inner_dim, 1, bias=False),
            nn.Conv1d(inner_dim, inner_dim, 3, bias=False, padding=1, groups=inner_dim)
        )

        self.to_k = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(dim, inner_dim, 1, bias=False),
            nn.Conv1d(inner_dim, inner_dim, 3, bias=False, padding=1, groups=inner_dim)
        )

        self.to_v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(dim, inner_dim, 1, bias=False),
            nn.Conv1d(inner_dim, inner_dim, 3, bias=False, padding=1, groups=inner_dim)
        )

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim),
                                        nn.Linear(context_dim, inner_dim * 2, bias=False)) if exists(
            context_dim) else None

        self.to_out = nn.Sequential(
            nn.Conv1d(inner_dim, dim, 1, bias=False),
            ChanLayerNorm(dim)
        )

    def forward(self, fmap, context=None):
        h, x, y = self.heads, *fmap.shape[-2:]

        fmap = self.norm(fmap)
        q, k, v = map(lambda fn: fn(fmap), (self.to_q, self.to_k, self.to_v))
        q, k, v = rearrange_many((q, k, v), 'b (h c) x y -> (b h) (x y) c', h=h)

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim=-1)
            ck, cv = rearrange_many((ck, cv), 'b n (h d) -> (b h) n d', h=h)
            k = torch.cat((k, ck), dim=-2)
            v = torch.cat((v, cv), dim=-2)

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h=h, x=x, y=y)

        out = self.nonlin(out)
        return self.to_out(out)


class GlobalContext(nn.Module):
    """ basically a superior form of squeeze-excitation that is attention-esque """

    def __init__(
            self,
            *,
            dim_in,
            dim_out
    ):
        super().__init__()
        self.to_k = nn.Conv1d(dim_in, 1, 1)
        hidden_dim = max(3, dim_out // 2)

        self.net = nn.Sequential(
            nn.Conv1d(dim_in, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, dim_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        context = self.to_k(x)
        x, context = rearrange_many((x, context), 'b n ... -> b n (...)')
        out = einsum('b i n, b c n -> b c i', context.softmax(dim=-1), x)

        return self.net(out)


def FeedForward(dim, mult=2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias=False),
        nn.GELU(),
        LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, dim, bias=False)
    )




def ChanFeedForward(dim,
                    mult=2):  # in paper, it seems for self attention layers they did feedforwards with twice channel width
    ChanLayerNorm = partial(LayerNorm, dim=-2)
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        ChanLayerNorm(dim),
        nn.Conv1d(dim, hidden_dim, 1, bias=False),
        nn.GELU(),
        ChanLayerNorm(hidden_dim),
        nn.Conv1d(hidden_dim, dim, 1, bias=False)
    )


class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            *,
            depth=1,
            heads=8,
            dim_head=32,
            ff_mult=2,
            context_dim=None,
            cosine_sim_attn=False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                EinopsToAndFrom('b c h', 'b h c',
                                Attention(dim=dim, heads=heads, dim_head=dim_head, context_dim=context_dim,
                                          cosine_sim_attn=cosine_sim_attn)),
                ChanFeedForward(dim=dim, mult=ff_mult)
            ]))

    def forward(self, x, context=None):
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x) + x
        return x


class LinearAttentionTransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            *,
            depth=1,
            heads=8,
            dim_head=32,
            ff_mult=2,
            context_dim=None,
            **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LinearAttention(dim=dim, heads=heads, dim_head=dim_head, context_dim=context_dim),
                ChanFeedForward(dim=dim, mult=ff_mult)
            ]))

    def forward(self, x, context=None):
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x) + x
        return x


class CrossEmbedLayer(nn.Module):
    def __init__(
            self,
            dim_in,
            kernel_sizes,
            dim_out=None,
            stride=2
    ):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv1d(dim_in, dim_scale, kernel, stride=stride, padding=(kernel - stride) // 2))

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim=1)


class UpsampleCombiner(nn.Module):
    def __init__(
            self,
            dim,
            *,
            enabled=False,
            dim_ins=tuple(),
            dim_outs=tuple()
    ):
        super().__init__()
        dim_outs = cast_tuple(dim_outs, len(dim_ins))
        assert len(dim_ins) == len(dim_outs)

        self.enabled = enabled

        if not self.enabled:
            self.dim_out = dim
            return

        self.fmap_convs = nn.ModuleList([Block(dim_in, dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)])
        self.dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)

    def forward(self, x, fmaps=None):
        target_size = x.shape[-1]

        fmaps = default(fmaps, tuple())

        if not self.enabled or len(fmaps) == 0 or len(self.fmap_convs) == 0:
            return x

        fmaps = [resize_image_to(fmap, target_size) for fmap in fmaps]
        outs = [conv(fmap) for fmap, conv in zip(fmaps, self.fmap_convs)]
        return torch.cat((x, *outs), dim=1)


# In[22]:


class Unet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lowres_cond = False
        self.dummy_parameter = nn.Parameter(torch.tensor([0.]))

    def cast_model_parameters(self, *args, **kwargs):
        return self

    def forward(self, x, *args, **kwargs):
        return x


class NullUnet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lowres_cond = False
        self.dummy_parameter = nn.Parameter(torch.tensor([0.]))

    def cast_model_parameters(self, *args, **kwargs):
        return self

    def forward(self, x, *args, **kwargs):
        return x



class attention(nn.Module):

    # k is the size of the embbeding
    def __init__(self, d_model=512 , att_dim=None, nhead=8):
        super().__init__()
        if att_dim == None:
            att_dim = int(d_model/nhead)
        self.d_model , self.nhead = d_model , nhead

        self.self_att_size = att_dim

        # These compute the queries, keys and values for all
        # nhead (as a single concatenated vector)
        # takes in k feat  and outputs K* number nhead with no bias
        ## these are where the W matrix come in
        self.tokeys = nn.Linear(d_model, self.self_att_size * nhead, bias=False)
        self.toqueries = nn.Linear(d_model , self.self_att_size * nhead, bias=False)
        self.tovalues = nn.Linear(d_model , att_dim * nhead, bias=False)

        # This unifies the outputs of the different heads into
        # a single k-vector
        # combines the info from all heads back to the orignial k
        self.unifyheads = nn.Linear(nhead * att_dim , d_model )

    def forward(self, q,k,v):
        # b =  batch
        # qs = number of query vectors
        # d_model is feature/embedding dim
        b, qs, d_model = q.size()
        b, ks, d_model = k.size()
        h = self.nhead

        # view is basically reshape
        # this applies the linear model defined above on input x
        # output  will be b
        queries = self.toqueries(q).view(b, qs, h, self.self_att_size)
        keys = self.tokeys(k).view(b, ks, h, self.self_att_size)
        values = self.tovalues(v).view(b, ks, h, self.self_att_size)

        #  Since the head and batch dimension are not next to each other
        #  we need to transpose before we reshape
        queries = queries.transpose(1, 2).contiguous().view(b * h, qs, self.self_att_size)
        keys = keys.transpose(1, 2).contiguous().view(b * h, ks, self.self_att_size)
        values = values.transpose(1, 2).contiguous().view(b * h, ks,self.self_att_size )

        # we move the scaling of the dot product by k−−√ back and instead scale the keys and queries by k−−√4 before multiplying them together
        # scalling
        queries = queries / (self.self_att_size ** (1 / 4))
        keys = keys / (self.self_att_size  ** (1 / 4))

        # TORCH.BMM
        # Performs a batch matrix-matrix product of matrices stored in input and mat2.
        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        # - dot has size (b*h, qs, t) containing raw weights

        dot = F.softmax(dot, dim=2)
        # - dot now contains row-wise normalized weights

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, qs, self.self_att_size)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, qs, h * self.self_att_size)
        return self.unifyheads(out)




class transformer_denoiser(nn.Module):
    def __init__(  self,
           #actual params
           encoderHeads = 8,
            decoderHeads=8,
           learned_sinu_pos_emb_dim=16,
            time_cond_dim = 64,
            internalDim = 64,
            encoderDepth = 2,
            decoderDepth = 2,
            #Unet overhead params
            lowres_cond=False,
            channels=4,
            channels_out=1,
            cond_on_text=True,
        text_embed_dim = None ):

        super().__init__()

        self._locals = locals()
        self._locals.pop('self', None)
        self._locals.pop('__class__', None)

        self.lowres_cond = lowres_cond

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1


        self.to_time_hiddens = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(sinu_pos_emb_input_dim, time_cond_dim),
            nn.SiLU()
        )


        self.linIn = nn.Linear( 4, internalDim)

        self.timeToCondDim = nn.Linear( 1,  internalDim)
        self.textToCondDim = nn.Linear( 2,  internalDim)

        self.linOut = nn.Linear(internalDim,4)

        self.encoder =   nn.ModuleList()
        for encodeMode in range(encoderDepth):
            self.encoder.append(nn.ModuleList(
                [attention(d_model=internalDim, nhead=encoderHeads),
                  nn.LayerNorm(internalDim),
                 nn.Sequential(
                    nn.Linear(internalDim,internalDim*4),
                    nn.ReLU(),
                    nn.Linear(4*internalDim,internalDim)),
                 nn.LayerNorm(internalDim),
                 ]))

        self.decoder =   nn.ModuleList()
        for decoder  in range(decoderDepth):
            self.decoder.append(nn.ModuleList(
                [attention(d_model=internalDim,nhead=decoderHeads),
                 nn.LayerNorm(internalDim),
                 attention(d_model=internalDim, nhead=decoderHeads),
                 nn.LayerNorm(internalDim),
                 nn.Sequential(
                    nn.Linear(internalDim,internalDim*4),
                    nn.ReLU(),
                    nn.Linear(4*internalDim,internalDim)),
                 nn.LayerNorm(internalDim),
                 ]))



        encoder_layer = nn.TransformerEncoderLayer(d_model=internalDim, nhead=encoderHeads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoderDepth)

        self.multihead_attn = nn.MultiheadAttention(internalDim, encoderHeads)
        # attn_output, attn_output_weights = multihead_attn(query, key, value)

        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=encoderDepth)






    # if the current settings for the unet are not correct
    # for cascading DDPM, then reinit the unet with the right settings
    def cast_model_parameters(
            self,
            *,
            lowres_cond,
            text_embed_dim,
            channels,
            channels_out,
            cond_on_text
    ):

        updated_kwargs = dict(
            lowres_cond=lowres_cond,
            text_embed_dim=text_embed_dim,
            channels=channels,
            channels_out=channels_out,
            cond_on_text=cond_on_text
        )

        return self.__class__(**{**self._locals, **updated_kwargs})


    def forward(
            self,
            x,
            time,
            *,
            lowres_cond_img=None,
            lowres_noise_times=None,
            text_embeds=None,
            text_mask=None,
            cond_images=None,
            self_cond=None,
            cond_drop_prob=0.
    ):



        batch_size, device = x.shape[0], x.device

        # TEMPORAL
        time_hiddens = self.to_time_hiddens(time)
        time_hiddens = time_hiddens.unsqueeze(2)
        time_hiddens = self.timeToCondDim(time_hiddens)

        # TEXT
        text_hiddens = self.textToCondDim(text_embeds)



        # COMBINE
        cond = torch.cat((time_hiddens,text_hiddens),1)


        x = torch.transpose(x, 1, 2)
        x = self.linIn(x)



        # ENCODER
        for att,norm1, ff, norm2 in self.encoder:
            cond = norm1(att(cond,cond,cond)+cond)
            cond = norm2(ff(cond))


        # DECODER
        for att1,norm1,att2,norm2, ff,norm3, in self.decoder:
            x = norm1(att1(x,x,x)+x)
            x = norm2(att2(x,cond,cond)+x)
            x = norm3(ff(x))


        #
        #
        # # x =  self.transformer_encoder(x)
        #
        # attn_output, attn_output_weights = self.multihead_attn(x, y, y)
        x = self.linOut(x)
        x = torch.transpose(x, 1, 2)

        return x