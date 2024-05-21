import os, sys
import time
import copy

import torch
from torch import nn
from nanoDataPrep  import  load_data_set_seq2seq
from utils import  once, eval_decorator, exists,resize_image_to, val_loop

from uNets_mod import  OneD_Unet, NullUnet
from myElucidatedImagen import  ElucidatedImagen
from ImagenTrainer import ImagenTrainer
from packaging import version

import numpy as np





def train_loop(model,
               train_loader,
               test_loader,
               optimizer=None,
               val_every=10,
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
               modelName = "vanUNet",
               device = 'cpu',
               ):
    if not exists(trainer):
        if not exists(optimizer):
            print("ERROR: If trainer not used, need to provide optimizer.")
    if exists(trainer):
        print("Trainer provided... will be used")
    steps = start_step


    baseFname =  "model_perf/" + modelName
    # useful if model is pretrained
    best_val_loss = val_loop(test_loader, model, device, train_unet_number)
    epochWithBestLoss = float('inf')

    start = time.time()
    for e in range(1, epochs + 1):

        loss_total = 0
        torch.cuda.empty_cache()
        print("######################################################################################")
        start = time.time()
        print("NOW: Training epoch: ", e + start_ep)

        # TRAINING
        samplesEnc = 0
        model.train()


        samplesPerEpoch=0
        print("Loop over ", len(train_loader), " batches (print . every ", val_every, " steps)")
        for num, item in enumerate(train_loader):
            X_train_batch = item[0].to(device)
            y_train_batch = item[1].to(device)

            samplesEnc += X_train_batch.shape[0]

            if exists(trainer):
                # this calls the foward method from the Imagen trainer
                loss = trainer(
                    X_train_batch, y_train_batch,
                    unet_number=train_unet_number,
                    max_batch_size=max_batch_size,
                    # auto divide the batch of 64 up into batch size of 4 and accumulate gradients, so it all fits in memory
                )
                trainer.update(unet_number=train_unet_number)

            else:
                optimizer.zero_grad()
                loss = model(X_train_batch, y_train_batch, unet_number=train_unet_number)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                optimizer.step()


            loss_total = loss_total + loss.item()
            avgLoss = loss_total/samplesEnc


            # writint to file
            trainLossF = open(baseFname + "_train_loss.csv", "a")
            timeF = open(baseFname + "_time.csv", "a")
            trainLossF.write( str(avgLoss) + ",")
            timeF.write( str(time.time() - start) + "," )
            trainLossF.close()
            timeF.close()


            if steps % val_every == 0:
                print( "loss on training = " + str(avgLoss), end="")
                val_loss = val_loop(test_loader, model, device, train_unet_number)
                print(" on the validation set total loss is "  +str(val_loss))
                valLossF = open(baseFname + "_val_loss.csv", "a")
                valLossF.write(str(val_loss) + ",")
                valLossF.close()

                if val_loss < best_val_loss:
                    epochWithBestLoss = e
                    print(" new best model on validation ")
                    best_val_loss = val_loss

                    # fname = "model_params/" +  modelName + "_trainer_save-model-epoch.pt"
                    # trainer.save(fname)
                    if save_model:
                        checkpoint = {'state_dict': model.state_dict(),
                                      'optimizer': trainer.optim0.state_dict()}
                        fname = "model_params/" + modelName + "_statedict_save-model-epoch.pt"
                        torch.save(checkpoint, fname)
                        print(f"Model saved: ")

                # early stopping  # if we haven't found a new best model in the last 100 steps
                if epochWithBestLoss < e - 10:
                    print(" model hasn't improved in 10 epochs returning ")
                    return

            steps = steps + 1

        print(f"\n\n-------------------\nTime for epoch {e}={(time.time() - start) / 60}\n-------------------")