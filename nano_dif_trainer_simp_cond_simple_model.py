#!/usr/bin/env python
# coding: utf-8

# # Model A: Protein Generator Diffusion model conditioned on overall secondary structure content
#
# ### Generative method to design novel proteins using a diffusion model
#
# B. Ni, D.L. Kaplan, M.J. Buehler, Generative design of de novo proteins based on secondary structure constraints using an attention-based diffusion model, Chem, 2023

# In[ ]:


import os, sys
import math

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # turn off CUDA if needed
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # select which GPU device is to be used

import shutil

# In[47]:
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

import matplotlib.pyplot as plt
import numpy as np

import os
import time
import copy

import torch
from torch import nn


# appending a path
sys.path.append('data_prep')
sys.path.append('model_classes')

from nanoDataPrep  import  load_data_set_seq2seq
from utils import  once, eval_decorator, exists,resize_image_to,val_loop
from train_set_up import  train_loop

from uNets_super_simple import  OneD_Unet, NullUnet
from myElucidatedImagen import  ElucidatedImagen
from ImagenTrainer import ImagenTrainer
from packaging import version

import numpy as np






def params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total parameters: ", pytorch_total_params, " trainable parameters: ", pytorch_total_params_trainable)




class ProteinDesigner_A(nn.Module):
    def __init__(self, timesteps=10, dim=32, pred_dim=25, loss_type=0, elucidated=False,
                 padding_idx=0,
                 cond_dim=512,
                 text_embed_dim=512,
                 input_tokens=25,  # for non-BERT
                 sequence_embed=False,
                 embed_dim_position=32,
                 max_seq_len=40,
                 max_text_len=16,
                 device='cuda:0',

                 ):
        super(ProteinDesigner_A, self).__init__()

        self.device = device
        self.pred_dim = pred_dim
        self.loss_type = loss_type

        self.fcBright = nn.Linear(1,max_text_len)
        self.fc_embed1 = nn.Linear(8, max_seq_len)  # NOT USED
        self.fc_embed2 = nn.Linear(1, text_embed_dim)  #
        self.max_text_len = max_text_len

        self.norm1 = nn.LayerNorm( self.max_text_len)

        # we will be blowing up the brightness dim
        final_max_len = 2*max_text_len

        # print("final cond len = " + str(final_max_len ))
        self.pos_emb_x = nn.Embedding(final_max_len + 1, embed_dim_position)
        text_embed_dim = text_embed_dim + embed_dim_position
        self.pos_matrix_i = torch.zeros(final_max_len, dtype=torch.long)
        for i in range(final_max_len):
            self.pos_matrix_i[i] = i + 1

        assert (loss_type == 0), "Loss other than MSE not implemented"

        unet1 = OneD_Unet(
            dim=dim,
            text_embed_dim=text_embed_dim,
            cond_dim=cond_dim,
            dim_mults=(1, 2),
            num_resnet_blocks=1,  # 1,
            layer_attns=(True,True),
            layer_cross_attns=(True, True),
            channels=self.pred_dim,
            channels_out=self.pred_dim,
            attn_dim_head=64,
            attn_heads=8,
            ff_mult=2.,
            lowres_cond=False,  # for cascading diffusion - https://cascaded-diffusion.github.io/
            layer_attns_depth=1,
            layer_attns_add_text_cond=True,
            # whether to condition the self-attention blocks with the text embeddings, as described in Appendix D.3.1
            attend_at_middle=True,
            # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
            use_linear_attn=False,
            use_linear_cross_attn=False,
            cond_on_text=True,
            max_text_len=final_max_len,
            init_dim=None,
            resnet_groups=8,
            init_conv_kernel_size=7,  # kernel size of initial conv, if not using cross embed
            init_cross_embed=False,  # TODO - fix ouput size calcs for conv1d
            init_cross_embed_kernel_sizes=(3, 7, 15),
            cross_embed_downsample=False,
            cross_embed_downsample_kernel_sizes=(2, 4),
            attn_pool_text=True,
            attn_pool_num_latents=32,  # 32,   #perceiver model latents
            dropout=0.,
            memory_efficient=False,
            init_conv_to_final_conv_residual=False,
            use_global_context_attn=True,
            scale_skip_connection=True,
            final_resnet_block=True,
            final_conv_kernel_size=3,
            cosine_sim_attn=True,
            self_cond=False,
            combine_upsample_fmaps=True,
            # combine feature maps from all upsample blocks, used in unet squared successfully
            pixel_shuffle_upsample=False,  # may address checkboard artifacts

        ).to(self.device)

        assert elucidated, "Only elucidated model implemented...."
        self.is_elucidated = elucidated
        if elucidated:
            self.imagen = ElucidatedImagen(
                unets=(unet1),
                channels=self.pred_dim,
                channels_out=self.pred_dim,
                loss_type=loss_type,
                text_embed_dim=text_embed_dim,
                image_sizes=[max_seq_len],
                cond_drop_prob=0.2,
                auto_normalize_img=False,
                num_sample_steps=timesteps,
                # (64, 32), # number of sample steps - 64 for base unet, 32 for upsampler (just an example, have no clue what the optimal values are)
                sigma_min=0.002,  # min noise level
                sigma_max=160,
                # (80, 160),       # max noise level, @crowsonkb recommends double the max noise level for upsampler
                sigma_data=0.5,  # standard deviation of data distribution
                rho=7,  # controls the sampling schedule
                P_mean=-1.2,  # mean of log-normal distribution from which noise is drawn for training
                P_std=1.2,  # standard deviation of log-normal distribution from which noise is drawn for training
                S_churn=40,
                # 80,                # parameters for stochastic sampling - depends on dataset, Table 5 in apper
                S_tmin=0.05,
                S_tmax=50,
                S_noise=1.003,

            ).to(self.device)
        else:
            print("Not implemented.")

    def forward(self, cond, output, unet_number=1):  # sequences=conditioning, output=prediction


        # extract the brightness
        brightness = cond[:,-1]
        logB = torch.log(brightness)
        logB = logB.unsqueeze(1)
        logB = self.fcBright(logB)

        cond = torch.cat((cond[:,:-1],logB),1)

        # creates an embedding for each position in the input
        cond = cond.unsqueeze(2)
        cond = self.fc_embed2(cond)

        pos_matrix_i_ = self.pos_matrix_i.repeat(cond.shape[0], 1).to(self.device)
        pos_emb_cond = self.pos_emb_x(pos_matrix_i_)
        pos_emb_cond = torch.squeeze(pos_emb_cond, 1)
        x = torch.cat((cond,  pos_emb_cond), 2)

        # here x is the conditiioning
        loss = self.imagen(output, text_embeds=x, unet_number=unet_number, )

        return loss

    def sample(self, cond, stop_at_unet_number=1, cond_scale=7.5, ):
        # extract the brightness
        brightness = cond[:,-1]
        logB = torch.log(brightness)
        logB = logB.unsqueeze(1)
        logB = self.fcBright(logB)

        cond = torch.cat((cond[:,:-1],logB),1)

        # creates an embedding for each position in the input
        cond = cond.unsqueeze(2)
        cond = self.fc_embed2(cond)

        pos_matrix_i_ = self.pos_matrix_i.repeat(cond.shape[0], 1).to(self.device)
        pos_emb_cond = self.pos_emb_x(pos_matrix_i_)
        pos_emb_cond = torch.squeeze(pos_emb_cond, 1)
        x = torch.cat((cond,  pos_emb_cond), 2)

        output = self.imagen.sample(text_embeds=x, cond_scale=cond_scale, stop_at_unet_number=stop_at_unet_number)

        return output


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    torch.rand(10).to(device)

    print(" hopefully running on ")
    print(device)

    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    num_of_gpus = torch.cuda.device_count()
    print(num_of_gpus)

    print("Torch version:", torch.__version__)

    ynormfac = 22.
    batch_size_ = 512
    batch_size_ = 32

    # i think max length needs to be a power of two for effectiv dowsample
    max_seq_len = 40 # 10 * 4 bases
    number = 99999999999999999
    min_length = 0
    file_path = 'PROTEIN_Mar18_2022_SECSTR_ALL.csv'
    file_path = "data/seq_with_dis.csv"
    train_loader, train_loader_noshuffle, test_loader, tokenizer_y = load_data_set_seq2seq(file_path=file_path,
                                                                                           min_length=0,
                                                                                           max_length= max_seq_len,
                                                                                           batch_size_=batch_size_,
                                                                                           output_dim=3,
                                                                                           maxdata=number,
                                                                                           split=0.1,
                                                                                           ynormfac = 22.)


    print_once = once(print)

    __version__ = '1.9.3'


    loss_list = []



    # need to cut in half to fit on GPU!

    pred_dim = 4
    cond_dim = 1
    dim = int((768)/2)
    dim = 256 + 128 # max



    # embed_dim_position = 12
    # pred_dim = 1
    # cond_dim = 5 + embed_dim_position
    # dim = 64

    # this needs to allign with the condition length
    max_text_len = 64
    # this will be doubled as half will go to the brightness
    # embed_dim_position = 128

    model_A = ProteinDesigner_A(timesteps=(96), dim=dim, pred_dim=pred_dim,
                                loss_type=0, elucidated=True,
                                padding_idx=0,
                                cond_dim=cond_dim,
                                text_embed_dim=cond_dim,
                                max_text_len=max_text_len,
                                device=device,
                                embed_dim_position=cond_dim
                                ).to(device)

    params(model_A)
    params(model_A.imagen.unets[0])

    train_unet_number = 1
    modelName = "simpCond"

    trainer = ImagenTrainer(model_A)

    loadModel = False
    if loadModel == True:
        fname = f"model_params/{ modelName }_statedict_save-model-epoch.pt"  # Early stopping checkpoint
        print("loading " + fname)
        checkpoint = torch.load(fname)
        model_A.load_state_dict(checkpoint['state_dict'])
        trainer.optim0.load_state_dict(checkpoint['optimizer'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # print(" using the pre trained model on the validation set total loss is "
        #       + str(val_loop(test_loader, model_A, device, train_unet_number)))




    train_loop(model_A,
               train_loader,
               test_loader,
               optimizer=None,
               val_every=10,
               epochs=2400,
               start_ep=0,
               start_step=0,
               train_unet_number=1,
               print_loss=50 * len(train_loader) - 1,
               trainer=trainer,
               plot_unscaled=False,  # if unscaled data is plotted
               max_batch_size=16,  # if trainer....
               save_model=True,
               cond_scales=[1.],
               num_samples=1, foldproteins=True,
               modelName  = modelName,
               device = device )



