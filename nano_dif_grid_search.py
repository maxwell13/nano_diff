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


from ImagenTrainer import ImagenTrainer
from packaging import version

import numpy as np



def params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total parameters: ", pytorch_total_params, " trainable parameters: ", pytorch_total_params_trainable)



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
    max_seq_len = 10 # 10 * 4 bases
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


    pred_dim = 4
    maxEpoch = 100
    max_text_len = 64
    save_model =False

    model_dims = [ 32,64,128,256,256 + 128]



    from nano_dif_trainer_simp_cond_transform_model import ProteinDesigner_A
    modelName = "transformerSimpModel"

    for depth in range(0,6):
        for dim in model_dims:
                saveFileName = modelName + "model_dim=" + str(dim) + "depth=" + str(depth)

                print(" going to run "  + saveFileName)

                model_A = ProteinDesigner_A(timesteps=(96), pred_dim=pred_dim,
                                            loss_type=0, elucidated=True,
                                            max_seq_len=10,
                                            max_text_len=max_text_len,
                                            device=device,
                                            # actual params
                                            encoderHeads=8,
                                            decoderHeads=8,
                                            learned_sinu_pos_emb_dim=16,
                                            time_cond_dim=64,
                                            internalDim=dim,
                                            encoderDepth=depth,
                                            decoderDepth=depth,
                                            ).to(device)

                params(model_A)
                params(model_A.imagen.unets[0])
                train_unet_number = 1
                trainer = ImagenTrainer(model_A)

                train_loop(model_A,
                           train_loader,
                           test_loader,
                           optimizer=None,
                           val_every=10,
                           epochs=maxEpoch,
                           start_ep=0,
                           start_step=0,
                           train_unet_number=1,
                           print_loss=50 * len(train_loader) - 1,
                           trainer=trainer,
                           plot_unscaled=False,  # if unscaled data is plotted
                           max_batch_size=16,  # if trainer....
                           save_model=save_model,
                           cond_scales=[1.],
                           num_samples=1, foldproteins=True,
                           modelName  = saveFileName ,
                           device = device )






    maxEpoch = 10


    models = [ "binnedCond","simpCond","noResNes"]
    # models = [ "noResNes"]




    cond_dims = [1,4,16,32,64]


    for modelName in models:
        if modelName == "noResNes":
            from nano_dif_trainer_simp_cond_simple_model import ProteinDesigner_A

        if modelName == "vanUNet":
            from nano_dif_trainer_van import ProteinDesigner_A
        if modelName == "binnedCond":
            from nano_dif_trainer_bind_cond import ProteinDesigner_A
        if modelName == "simpCond":
            from nano_dif_trainer_simp_cond import ProteinDesigner_A


        for dim in model_dims:
            for cond_dim in cond_dims:
                saveFileName = modelName + "model_dim=" + str(dim) + "cond_dim=" + str(cond_dim)

                print(" going to run "  + saveFileName)

                model_A = ProteinDesigner_A(timesteps=(96), dim=dim, pred_dim=pred_dim,
                                            loss_type=0, elucidated=True,
                                            padding_idx=0,
                                            cond_dim=cond_dim,
                                            text_embed_dim=cond_dim,
                                            max_text_len=max_text_len,
                                            device=device,
                                            embed_dim_position=cond_dim,
                                            max_seq_len = max_seq_len
                                            ).to(device)

                params(model_A)
                params(model_A.imagen.unets[0])
                train_unet_number = 1
                trainer = ImagenTrainer(model_A)

                train_loop(model_A,
                           train_loader,
                           test_loader,
                           optimizer=None,
                           val_every=10,
                           epochs=maxEpoch,
                           start_ep=0,
                           start_step=0,
                           train_unet_number=1,
                           print_loss=50 * len(train_loader) - 1,
                           trainer=trainer,
                           plot_unscaled=False,  # if unscaled data is plotted
                           max_batch_size=16,  # if trainer....
                           save_model=save_model,
                           cond_scales=[1.],
                           num_samples=1, foldproteins=True,
                           modelName  = saveFileName ,
                           device = device )



