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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # turn off CUDA if needed
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # select which GPU device is to be used

import shutil

from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.preprocessing.text import Tokenizer

# In[47]:
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB import PDBList

# In[2]:


import torch
# from imagen_pytorch import Unet, Imagen

import matplotlib.pyplot as plt
import numpy as np




import os
import time
import copy
from pathlib import Path

from contextlib import contextmanager, nullcontext
from functools import partial, wraps


import torch
from torch import nn
import torch.nn.functional as F

# appending a path
sys.path.append('data_prep')

from dataPrep  import  load_data_set_seq2seq
from utils import  once, eval_decorator, exists,resize_image_to

from uNets import  OneD_Unet, NullUnet
from ElucidatedImagen import  ElucidatedImagen
from ImagenTrainer import ImagenTrainer
from packaging import version

import numpy as np




device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
num_of_gpus = torch.cuda.device_count()
print(num_of_gpus)

print("Torch version:", torch.__version__)


def params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total parameters: ", pytorch_total_params, " trainable parameters: ", pytorch_total_params_trainable)


# In[13]:
# In[15]:


ynormfac = 22.
batch_size_ = 512
batch_size_ = 1
max_length = 64
number = 99999999999999999
min_length = 0
file_path = 'PROTEIN_Mar18_2022_SECSTR_ALL.csv'
file_path = "data/fake_data.csv"
train_loader, train_loader_noshuffle, test_loader, tokenizer_y = load_data_set_seq2seq(file_path=file_path,
                                                                                       min_length=0,
                                                                                       max_length=max_length,
                                                                                       batch_size_=batch_size_,
                                                                                       output_dim=3,
                                                                                       maxdata=number,
                                                                                       split=0.1,
                                                                                       ynormfac = 22.)




print_once = once(print)




# predefined unets, with configs lining up with hyperparameters in appendix of paper


# In[25]:


########################################################
## Elucidated denoising model
## After: Tero Karras and Miika Aittala and Timo Aila and Samuli Laine,
##        Elucidating the Design Space of Diffusion-Based Generative Models
##        https://arxiv.org/abs/2206.00364, 2022
########################################################


# constants








# In[68]:


class ProteinDesigner_A(nn.Module):
    def __init__(self, timesteps=10, dim=32, pred_dim=25, loss_type=0, elucidated=False,
                 padding_idx=0,
                 cond_dim=512,
                 text_embed_dim=512,
                 input_tokens=25,  # for non-BERT
                 sequence_embed=False,
                 embed_dim_position=32,
                 max_text_len=16,
                 device='cuda:0',

                 ):
        super(ProteinDesigner_A, self).__init__()

        self.device = device
        self.pred_dim = pred_dim
        self.loss_type = loss_type

        self.fc_embed1 = nn.Linear(8, max_length)  # NOT USED
        self.fc_embed2 = nn.Linear(1, text_embed_dim)  #
        self.max_text_len = max_text_len

        self.pos_emb_x = nn.Embedding(max_text_len + 1, embed_dim_position)
        text_embed_dim = text_embed_dim + embed_dim_position
        self.pos_matrix_i = torch.zeros(max_text_len, dtype=torch.long)
        for i in range(max_text_len):
            self.pos_matrix_i[i] = i + 1

        assert (loss_type == 0), "Loss other than MSE not implemented"

        unet1 = OneD_Unet(
            dim=dim,
            text_embed_dim=text_embed_dim,
            cond_dim=cond_dim,
            dim_mults=(1, 2, 4, 8),

            num_resnet_blocks=1,  # 1,
            layer_attns=(False, True, True, False),
            layer_cross_attns=(False, True, True, False),
            channels=self.pred_dim,
            channels_out=self.pred_dim,
            #
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
            max_text_len=max_length,
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
                image_sizes=[max_length],
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

    def forward(self, x, output, unet_number=1):  # sequences=conditioning, output=prediction

        x = x.unsqueeze(2)

        x = self.fc_embed2(x)

        pos_matrix_i_ = self.pos_matrix_i.repeat(x.shape[0], 1).to(self.device)
        pos_emb_x = self.pos_emb_x(pos_matrix_i_)
        pos_emb_x = torch.squeeze(pos_emb_x, 1)
        pos_emb_x[:, x.shape[1]:, :] = 0  # set all to zero that are not provided via x
        pos_emb_x = pos_emb_x[:, :x.shape[1], :]
        x = torch.cat((x, pos_emb_x), 2)

        # here x is the conditiioning
        loss = self.imagen(output, text_embeds=x, unet_number=unet_number, )

        return loss

    def sample(self, x, stop_at_unet_number=1, cond_scale=7.5, ):

        x = x.unsqueeze(2)

        x = self.fc_embed2(x)

        pos_matrix_i_ = self.pos_matrix_i.repeat(x.shape[0], 1).to(self.device)
        pos_emb_x = self.pos_emb_x(pos_matrix_i_)
        pos_emb_x = torch.squeeze(pos_emb_x, 1)
        pos_emb_x[:, x.shape[1]:, :] = 0  # set all to zero that are not provided via x
        pos_emb_x = pos_emb_x[:, :x.shape[1], :]
        x = torch.cat((x, pos_emb_x), 2)

        output = self.imagen.sample(text_embeds=x, cond_scale=cond_scale, stop_at_unet_number=stop_at_unet_number)

        return output

    # ## Training loop


# In[27]:




from torch.cuda.amp import autocast, GradScaler



__version__ = '1.9.3'



# helper functions


def write_fasta(sequence, filename_out):
    with open(filename_out, mode='w') as f:
        f.write(f'>{filename_out}\n')
        f.write(f'{sequence}')


# In[29]:


def sample_sequence(model,
                    X=[[0.92, 0., 0.04, 0.04, 0., 0., 0., 0., ]],
                    flag=0,
                    cond_scales=1., foldproteins=False,
                    normalize_input_to_one=False,
                    calc_error=False,
                    ):
    steps = 0
    e = flag

    print(f"Producing {len(X)} samples...")

    print('Device: ', device)

    for iisample in range(len(X)):

        X_cond = torch.Tensor(X[iisample]).to(device).unsqueeze(0)

        if normalize_input_to_one:
            X_cond = X_cond / X_cond.sum()

        print("Conditoning used: ", X_cond, "...sum: ", X_cond.sum(), "cond scale: ", cond_scales)

        result = model.sample(X_cond, stop_at_unet_number=train_unet_number,
                              cond_scale=cond_scales)

        result = torch.round(result * ynormfac)
        plt.plot(result[0, 0, :].cpu().detach().numpy(), label=f'Predicted')

        plt.legend()

        outname = prefix + f"sampld_from_X_{flag}_condscale-{str(cond_scales)}_{e}_{steps}.jpg"

        # plt.savefig(outname, dpi=200)
        # plt.show ()

        to_rev = result[:, 0, :]
        to_rev = to_rev.long().cpu().detach().numpy()

        y_data_reversed = tokenizer_y.sequences_to_texts(to_rev)

        for iii in range(len(y_data_reversed)):
            y_data_reversed[iii] = y_data_reversed[iii].upper().strip().replace(" ", "")

        print(f"For {X}, predicted sequence", y_data_reversed[0])

        if foldproteins:
            xbc = X_cond[iisample, :].cpu().detach().numpy()
            out_nam = np.array2string(xbc, formatter={'float_kind': lambda xbc: "%.2f" % xbc}) + f'_{flag}_{steps}'
            tempname = 'temp'
            pdb_file = foldandsavePDB(sequence=y_data_reversed[0],
                                      filename_out=tempname,
                                      num_cycle=16, flag=flag)
            out_nam_fasta = f'{prefix}{out_nam}.fasta'

            out_nam = f'{prefix}{out_nam}.pdb'

            write_fasta(y_data_reversed[0], out_nam_fasta)

            shutil.copy(pdb_file, out_nam)  # source, dest
            pdb_file = out_nam
            print(f"Properly named PDB file produced: {pdb_file}")

            # view=show_pdb(pdb_file=pdb_file, flag=flag,
            #               show_sidechains=show_sidechains, show_mainchains=show_mainchains, color=color)
            # view.show()

        if calc_error:
            get_Model_A_error(pdb_file, X[iisample], plotit=True)
        return pdb_file


# In[51]:


def sample_loop(model,
                train_loader,
                cond_scales=[1.0],  # list of cond scales - each sampled...
                num_samples=2,  # how many samples produced every time tested.....
                timesteps=100,
                flag=0, foldproteins=False,
                calc_error=False,
                ):
    steps = 0
    e = flag
    for item in train_loader:

        X_train_batch = item[0].to(device)
        y_train_batch = item[1].to(device)

        GT = y_train_batch.cpu().detach()

        GT = resize_image_to(
            GT.unsqueeze(1),

            model.imagen.image_sizes[train_unet_number - 1],

        )

        num_samples = min(num_samples, y_train_batch.shape[0])
        print(f"Producing {num_samples} samples...")

        for iisample in range(len(cond_scales)):

            result = model.sample(X_train_batch, stop_at_unet_number=train_unet_number,
                                  cond_scale=cond_scales[iisample])

            result = torch.round(result * ynormfac)

            GT = torch.round(GT * ynormfac)
            for samples in range(num_samples):
                print("sample ", samples, "out of ", num_samples)

                plt.plot(result[samples, 0, :].cpu().detach().numpy(), label=f'Predicted')
                plt.plot(GT[samples, 0, :], label=f'GT {0}')
                plt.legend()

                outname = prefix + f"sample-{samples}_condscale-{str(cond_scales[iisample])}_{e}_{steps}.jpg"

                # plt.savefig(outname, dpi=200)
                # plt.show ()
                #
                to_rev = result[:, 0, :]
                to_rev = to_rev.long().cpu().detach().numpy()

                y_data_reversed = tokenizer_y.sequences_to_texts(to_rev)

                for iii in range(len(y_data_reversed)):
                    y_data_reversed[iii] = y_data_reversed[iii].upper().strip().replace(" ", "")

                print(f"For {X_train_batch[samples, :].cpu().detach().numpy()}, predicted sequence",
                      y_data_reversed[samples])
                if foldproteins:
                    xbc = X_train_batch[samples, :].cpu().detach().numpy()
                    out_nam = np.array2string(xbc,
                                              formatter={'float_kind': lambda xbc: "%.1f" % xbc}) + f'_{flag}_{samples}'
                    tempname = 'temp'
                    pdb_file = foldandsavePDB(sequence=y_data_reversed[samples],
                                              filename_out=tempname,
                                              num_cycle=16, flag=flag)
                    out_nam = f'{prefix}{out_nam}.pdb'
                    print(f'Original PDB: {pdb_file} OUT: {out_nam}')
                    shutil.copy(pdb_file, out_nam)  # source, dest
                    pdb_file = out_nam
                    print(f"Properly named PDB file produced: {pdb_file}")
                    #
                    # view=show_pdb(pdb_file=pdb_file, flag=flag,
                    #               show_sidechains=show_sidechains, show_mainchains=show_mainchains, color=color)
                    # view.show()

                steps = steps + 1

                if calc_error:
                    cond = X_train_batch[samples, :].cpu().detach().numpy()  # X_train_batch[samples,:]
                    get_Model_A_error(pdb_file, cond, plotit=True)

        if steps > num_samples:
            break


# In[31]:


def train_loop(model,
               train_loader,
               test_loader,
               optimizer=None,
               print_every=10,
               epochs=300,
               start_ep=0,
               start_step=0,
               train_unet_number=1,
               print_loss=1000,
               trainer=None,
               plot_unscaled=False,
               max_batch_size=4,
               save_model=False,
               cond_scales=[1.0],  # list of cond scales
               num_samples=2,  # how many samples produced every time tested.....
               foldproteins=False,
               ):
    if not exists(trainer):
        if not exists(optimizer):
            print("ERROR: If trainer not used, need to provide optimizer.")
    if exists(trainer):
        print("Trainer provided... will be used")
    steps = start_step

    loss_total = 0
    for e in range(1, epochs + 1):
        start = time.time()

        torch.cuda.empty_cache()
        print("######################################################################################")
        start = time.time()
        print("NOW: Training epoch: ", e + start_ep)

        # TRAINING
        train_epoch_loss = 0
        model.train()

        print("Loop over ", len(train_loader), " batches (print . every ", print_every, " steps)")
        for num, item in enumerate(train_loader):
            X_train_batch = item[0].to(device)
            y_train_batch = item[1].to(device)

            if exists(trainer):
                # this calls the foward method from the Imagen trainer
                loss = trainer(
                    X_train_batch, y_train_batch.unsqueeze(1),
                    unet_number=train_unet_number,
                    max_batch_size=max_batch_size,
                    # auto divide the batch of 64 up into batch size of 4 and accumulate gradients, so it all fits in memory
                )
                trainer.update(unet_number=train_unet_number)

            else:
                optimizer.zero_grad()
                loss = model(X_train_batch, y_train_batch.unsqueeze(1), unet_number=train_unet_number)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                optimizer.step()

            loss_total = loss_total + loss.item()

            if steps % print_every == 0:
                print(".", end="")

            if steps > 0:
                if steps % print_loss == 0:
                    # if plot_unscaled:
                    #
                    #     plt.plot (y_train_batch.unsqueeze(1)[0,0,:].cpu().detach().numpy(),label= 'Unscaled GT')
                    #     plt.legend()
                    #     plt.show()
                    #
                    # rescale GT to properly plot
                    GT = y_train_batch.cpu().detach()

                    GT = resize_image_to(
                        GT.unsqueeze(1),
                        model.imagen.image_sizes[train_unet_number - 1],

                    )

                    norm_loss = loss_total / print_loss
                    print(f"\nTOTAL LOSS at epoch={e}, step={steps}: {norm_loss}")

                    loss_list.append(norm_loss)
                    loss_total = 0

                    plt.plot(loss_list, label='Loss')
                    plt.legend()

                    outname = prefix + f"loss_{e}_{steps}.jpg"
                    # plt.savefig(outname, dpi=200)
                    # plt.show()

                    num_samples = min(num_samples, y_train_batch.shape[0])
                    print(f"Producing {num_samples} samples...")

                    sample_loop(model,
                                test_loader,
                                cond_scales=cond_scales,
                                num_samples=1,  # how many samples produced every time tested.....
                                timesteps=64,
                                flag=steps, foldproteins=foldproteins,
                                )

                    print("SAMPLING FOR DE NOVO:")
                    sample_sequence(model,
                                    X=[[0, 0.7, 0.07, 0.1, 0.01, 0.02, 0.01, 0.11]], foldproteins=foldproteins,
                                    flag=steps, cond_scales=1.,
                                    )
                    sample_sequence(model,
                                    X=[[0., 0.0, 0.0, 0.0, 0., 0., 0., 0., ]], foldproteins=foldproteins,
                                    flag=steps, cond_scales=1.,
                                    )

            if steps > 0:
                if save_model and steps % print_loss == 0:
                    fname = f"{prefix}trainer_save-model-epoch_{e}.pt"
                    trainer.save(fname)
                    fname = f"{prefix}statedict_save-model-epoch_{e}.pt"
                    torch.save(model.state_dict(), fname)
                    print(f"Model saved: ")

            steps = steps + 1

        print(f"\n\n-------------------\nTime for epoch {e}={(time.time() - start) / 60}\n-------------------")


# In[70]:


# pip install py3Dmol
# poymol vis: https://gist.github.com/bougui505/11401240

def foldandsavePDB(sequence, filename_out, num_cycle=16, flag=0):
    filename = f"{prefix}fasta_in_{flag}.fasta"
    print("Writing FASTA file: ", filename)
    OUTFILE = f"{filename_out}_{flag}"
    with open(filename, mode='w') as f:
        f.write(f'>{OUTFILE}\n')
        f.write(f'{sequence}')

    print("Now run OmegaFold....")
    # get_ipython().system('omegafold $filename $prefix --num_cycle $num_cycle --device=$device')
    print("Done OmegaFold")

    PDB_result = f"{prefix}{OUTFILE}.pdb"
    print(f"Resulting PDB file...:  {PDB_result}")

    return PDB_result


import py3Dmol


def plot_plddt_legend(dpi=100):
    thresh = ['plDDT:', 'Very low (<50)', 'Low (60)', 'OK (70)', 'Confident (80)', 'Very high (>90)']
    plt.figure(figsize=(1, 0.1), dpi=dpi)

    for c in ["#FFFFFF", "#FF0000", "#FFFF00", "#00FF00", "#00FFFF", "#0000FF"]:
        plt.bar(0, 0, color=c)
    plt.legend(thresh, frameon=False,
               loc='center', ncol=6,
               handletextpad=1,
               columnspacing=1,
               markerscale=0.5, )
    plt.axis(False)
    return plt


color = "lDDT"  # @param ["chain", "lDDT", "rainbow"]
show_sidechains = False  # @param {type:"boolean"}
show_mainchains = False  # @param {type:"boolean"}


def show_pdb(pdb_file, flag=0, show_sidechains=False, show_mainchains=False, color="lDDT"):
    model_name = f"Flag_{flag}"
    view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js', )
    view.addModel(open(pdb_file, 'r').read(), 'pdb')

    if color == "lDDT":
        view.setStyle({'cartoon': {'colorscheme': {'prop': 'b', 'gradient': 'roygb', 'min': 50, 'max': 90}}})
    elif color == "rainbow":
        view.setStyle({'cartoon': {'color': 'spectrum'}})
    elif color == "chain":
        chains = len(queries[0][1]) + 1 if is_complex else 1
        for n, chain, color in zip(range(chains), list("ABCDEFGH"),
                                   ["lime", "cyan", "magenta", "yellow", "salmon", "white", "blue", "orange"]):
            view.setStyle({'chain': chain}, {'cartoon': {'color': color}})
    if show_sidechains:
        BB = ['C', 'O', 'N']
        view.addStyle({'and': [{'resn': ["GLY", "PRO"], 'invert': True}, {'atom': BB, 'invert': True}]},
                      {'stick': {'colorscheme': f"WhiteCarbon", 'radius': 0.3}})
        view.addStyle({'and': [{'resn': "GLY"}, {'atom': 'CA'}]},
                      {'sphere': {'colorscheme': f"WhiteCarbon", 'radius': 0.3}})
        view.addStyle({'and': [{'resn': "PRO"}, {'atom': ['C', 'O'], 'invert': True}]},
                      {'stick': {'colorscheme': f"WhiteCarbon", 'radius': 0.3}})
    if show_mainchains:
        BB = ['C', 'O', 'N', 'CA']
        view.addStyle({'atom': BB}, {'stick': {'colorscheme': f"WhiteCarbon", 'radius': 0.3}})

    view.zoomTo()
    if color == "lDDT":
        plot_plddt_legend().show()
    return view


def get_avg_Bfac(file='./output_v3/[0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0].pdb'):
    p = PDBParser()
    avg_B = 0
    bfac_list = []

    structure = p.get_structure("X", file)
    for PDBmodel in structure:
        for chain in PDBmodel:
            for residue in chain:
                for atom in residue:
                    Bfac = atom.get_bfactor()
                    bfac_list.append(Bfac)
                    avg_B = avg_B + Bfac

    avg_B = avg_B / len(bfac_list)
    print(f"For {file}, average B-factor={avg_B}")
    plt.plot(bfac_list, label='lDDT')
    plt.xlabel('Atom #')
    plt.ylabel('iDDT')
    plt.legend()
    plt.show()
    return avg_B, bfac_list


def sample_sequence_normalized_Bfac(seccs=[0.3, 0.3, 0.1, 0., 0., 0., 0., 0.]):
    sample_numbers = torch.tensor([seccs])
    sample_numbers = torch.nn.functional.normalize(sample_numbers, dim=1)
    sample_numbers = sample_numbers / torch.sum(sample_numbers)

    print(torch.sum(sample_numbers[0]), sample_numbers)
    PDB = sample_sequence(model,
                          X=sample_numbers,
                          flag=0, cond_scales=1, foldproteins=True,
                          )

    avg, _ = get_avg_Bfac(file=PDB[0])

    return PDB, avg


# ### DDSP analysis and error calculation

# In[44]:


def get_DSSP_result(fname):
    pdb_list = [fname]

    # parse structure
    p = PDBParser()
    for i in pdb_list:
        structure = p.get_structure(i, fname)
        # use only the first model
        model = structure[0]
        # calculate DSSP
        dssp = DSSP(model, fname, file_type='PDB')
        # extract sequence and secondary structure from the DSSP tuple
        sequence = ''
        sec_structure = ''
        for z in range(len(dssp)):
            a_key = list(dssp.keys())[z]
            sequence += dssp[a_key][1]
            sec_structure += dssp[a_key][2]

        # print(i)
        # print(sequence)
        # print(sec_structure)
        #
        # The DSSP codes for secondary structure used here are:
        # =====     ====
        # Code      Structure
        # =====     ====
        # H         Alpha helix (4-12)
        # B         Isolated beta-bridge residue
        # E         Strand
        # G         3-10 helix
        # I         Pi helix
        # T         Turn
        # S         Bend
        # ~         None
        # =====     ====
        #

        sec_structure = sec_structure.replace('-', '~')
        sec_structure_3state = sec_structure

        # if desired, convert DSSP's 8-state assignments into 3-state [C - coil, E - extended (beta-strand), H - helix]
        sec_structure_3state = sec_structure_3state.replace('H', 'H')  # 0
        sec_structure_3state = sec_structure_3state.replace('E', 'E')
        sec_structure_3state = sec_structure_3state.replace('T', '~')
        sec_structure_3state = sec_structure_3state.replace('~', '~')
        sec_structure_3state = sec_structure_3state.replace('B', 'E')
        sec_structure_3state = sec_structure_3state.replace('G', 'H')  # 5
        sec_structure_3state = sec_structure_3state.replace('I', 'H')  # 6
        sec_structure_3state = sec_structure_3state.replace('S', '~')
        return sec_structure, sec_structure_3state, sequence


def string_diff(seq1, seq2):
    return sum(1 for a, b in zip(seq1, seq2) if a != b) + abs(len(seq1) - len(seq2))


# In[57]:


def get_Model_A_error(fname, cond, plotit=True, ploterror=False):
    sec_structure, sec_structure_3state, sequence = get_DSSP_result(fname)
    sscount = []
    length = len(sec_structure)
    sscount.append(sec_structure.count('H') / length)
    sscount.append(sec_structure.count('E') / length)
    sscount.append(sec_structure.count('T') / length)
    sscount.append(sec_structure.count('~') / length)
    sscount.append(sec_structure.count('B') / length)
    sscount.append(sec_structure.count('G') / length)
    sscount.append(sec_structure.count('I') / length)
    sscount.append(sec_structure.count('S') / length)
    sscount = np.asarray(sscount)

    error = np.abs(sscount - cond)
    print("Abs error per SS structure type (H, E, T, ~, B, G, I S): ", error)

    if ploterror:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        plt.plot(error, 'o-', label='Error over SS type')
        plt.legend()
        plt.ylabel('SS content')
        plt.show()

    x = np.linspace(0, 7, 8)

    sslabels = ['H', 'E', 'T', '~', 'B', 'G', 'I', 'S']

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))

    ax.bar(x - 0.15, cond, width=0.3, color='b', align='center')
    ax.bar(x + 0.15, sscount, width=0.3, color='r', align='center')

    ax.set_ylim([0, 1])

    plt.xticks(range(len(sslabels)), sslabels, size='medium')
    plt.legend(['GT', 'Prediction'])

    plt.ylabel('SS content')
    plt.show()

    ######################## 3 types

    sscount = []
    length = len(sec_structure)
    sscount.append(sec_structure_3state.count('H') / length)
    sscount.append(sec_structure_3state.count('E') / length)
    sscount.append(sec_structure_3state.count('~') / length)
    cond_p = [np.sum([cond[0], cond[5], cond[6]]), np.sum([cond[1], cond[4]]), np.sum([cond[2], cond[3], cond[7]])]

    print("cond 3type: ", cond_p)
    sscount = np.asarray(sscount)

    error3 = np.abs(sscount - cond_p)
    print("Abs error per 3-type SS structure type (C, H, E): ", error)

    if ploterror:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))

        plt.plot(error3, 'o-', label='Error over SS type')
        plt.legend()
        plt.ylabel('SS content')
        plt.show()

    x = np.linspace(0, 2, 3)

    sslabels = ['H', 'E', '~']

    # ax = plt.subplot(111, figsize=(4,4))
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))

    ax.bar(x - 0.15, cond_p, width=0.3, color='b', align='center')
    ax.bar(x + 0.15, sscount, width=0.3, color='r', align='center')

    ax.set_ylim([0, 1])

    plt.xticks(range(len(sslabels)), sslabels, size='medium')
    plt.legend(['GT', 'Prediction'])

    plt.ylabel('SS content')
    plt.show()

    return error


# ## Define model and train or load weights, inference, etc.

# In[35]:


loss_list = []

# In[37]:


prefix = './output_model_A/'
if not os.path.exists(prefix):
    os.mkdir(prefix)

# In[38]:


embed_dim_position = 128
pred_dim = 1
cond_dim = 512
dim = 768

embed_dim_position = 12
pred_dim = 1
cond_dim = 5 + embed_dim_position
dim = 64

model_A = ProteinDesigner_A(timesteps=(96), dim=dim, pred_dim=pred_dim,
                            loss_type=0, elucidated=True,
                            padding_idx=0,
                            cond_dim=cond_dim,
                            text_embed_dim=cond_dim - embed_dim_position,
                            embed_dim_position=embed_dim_position,
                            max_text_len=8,
                            device=device,
                            ).to(device)

params(model_A)
params(model_A.imagen.unets[0])

# In[39]:


train_unet_number = 1

# In[63]:


train_model = True  # do not train if false

if train_model:

    trainer = ImagenTrainer(model_A)
    train_loop(model_A,
               train_loader,
               test_loader,
               optimizer=None,
               print_every=100,
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
               )

else:
    # fname=f"{prefix}model_A_final.pt"  #Final checkpoint
    fname = f"{prefix}model_A_early.pt"  # Early stopping checkpoint
    model_A.load_state_dict(torch.load(fname))

# ### Generative examples

# In[64]:


sample_loop(model_A,
            test_loader,
            cond_scales=[1.],  # list of cond scales - each sampled...
            num_samples=4,  # how many samples produced every time tested.....
            timesteps=96, flag=10000, foldproteins=True,
            calc_error=True)

# In[65]:


norm_flag = False
flag_ref = 40000

sample_sequence(model_A,
                X=[[0, 0.7, 0.07, 0.1, 0.01, 0.02, 0.01, 0.11]],
                normalize_input_to_one=norm_flag,
                flag=flag_ref, cond_scales=1., foldproteins=True, calc_error=True,
                )

# In[66]:


#### Generate candidates

norm_flag = False
flag_ref = 40000

sample_sequence(model_A,
                X=[[0, 0.7, 0.07, 0.1, 0.01, 0.02, 0.01, 0.11]],
                normalize_input_to_one=norm_flag,
                flag=flag_ref, cond_scales=1., foldproteins=True, calc_error=True,
                )
sample_sequence(model_A,
                X=[[0.2, 0.2, 0.07, 0.3, 0.01, 0.02, 0.01, 0.11]],
                normalize_input_to_one=norm_flag,
                flag=flag_ref + 1, cond_scales=1., foldproteins=True, calc_error=True,
                )

sample_sequence(model_A,
                X=[[0.8, 0.0, 0.0, 0., 0.0, 0.0, 0.0, 0.]],
                normalize_input_to_one=norm_flag,
                flag=flag_ref + 2, cond_scales=1., foldproteins=True, calc_error=True,
                )

sample_sequence(model_A,
                X=[[0.5, 0.0, 0.0, 0.2, 0.0, 0.0, 0.1, 0.1]],
                normalize_input_to_one=norm_flag,
                flag=flag_ref + 3, cond_scales=1., foldproteins=True, calc_error=True,
                )

sample_sequence(model_A,
                X=[[0.01, 0.1, 0.0, 0.5, 0.0, 0.01, 0.1, 0.5]],
                normalize_input_to_one=norm_flag,
                flag=flag_ref + 4, cond_scales=1., foldproteins=True, calc_error=True,
                )

