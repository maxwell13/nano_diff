



from data_prep.nanoDataPrep  import  load_data_set_seq2seq

import torch
import  pandas as pd
import  numpy as np



device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")

def sample_loop(model,
                dist,
                cond_scales=[1.0],  # list of cond scales - each sampled...
                num_samples=2,  # how many samples produced every time tested.....
                timesteps=100,
                flag=0, foldproteins=False,
                calc_error=False,
                device="cpu"
                ):
    DNASeqs = []
    X_train_batch =  torch.from_numpy(dist).unsqueeze(0).to(torch.double)
    X_train_batch =X_train_batch.to(torch.float32).to(device)
    charToInt = {"A": 1, "C": 2, "G": 3, "T": 4}
    intToChar = dict((v, k) for k, v in charToInt.items())

    train_unet_number = 1
    # max one hot enocded dna

    iisample = 0

    for numSamps  in range(0,100):
        result = model.sample(X_train_batch, stop_at_unet_number=  train_unet_number,
                              cond_scale=cond_scales[iisample])

        for ind  in range( result.shape[0]):
            samp = result[ind,:,:]
            sampResSeq = ""
            [_,totSize]= samp.shape
            for secS in range(0,totSize,4):
                sec = samp[:,secS:secS+4]
                pos = np.argmax(sec.cpu().numpy())
                let =  intToChar[pos+1]
                sampResSeq +=let
        DNASeqs.append( sampResSeq )

    return DNASeqs

    print("done sampling ")












''' 
////////////////////////////////////////////////////////////////////////////////////////
MODEL Set up 
'''
available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
num_of_gpus = torch.cuda.device_count()
print(num_of_gpus)
print("Torch version:", torch.__version__)

max_length = 40
# need to cut in half to fit on GPU!
embed_dim_position = 64
pred_dim = 1
cond_dim = 256
dim = int((768)/2)
# embed_dim_position = 12
# pred_dim = 1
# cond_dim = 5 + embed_dim_position
# dim = 64

# this needs to allign with the condition length
max_text_len = 64


# load trained params
modelName  = "vanUNet"
# modelName = "simpCond"
modelName = "binnedCond"

if modelName == "vanUNet":
    from nano_dif_trainer_van import ProteinDesigner_A
if modelName == "binnedCond":
    pred_dim = 1
    cond_dim = 8
    dim = int((768)/2)
    from nano_dif_trainer_bind_cond import ProteinDesigner_A
if modelName == "simpCond":
    pred_dim = 1
    cond_dim = 8
    dim = int((768)/2)
    from nano_dif_trainer_simp_cond import ProteinDesigner_A

model_A = ProteinDesigner_A(timesteps=(96), dim=dim, pred_dim=pred_dim,
                            loss_type=0, elucidated=True,
                            padding_idx=0,
                            cond_dim=cond_dim,
                            text_embed_dim=cond_dim,
                            max_text_len=max_text_len,
                            device=device,
                            ).to(device)


# fname = "model_params/" + modelName + "_trainer_save-model-epoch.pt"
fname = "model_params/" + modelName + "_statedict_save-model-epoch.pt"
print("loading " + fname)
checkpoint = torch.load(fname,map_location=device)
model_A.load_state_dict(checkpoint['state_dict'])


''' 
Sampling 
'''

file_path = "data/seq_with_dis.csv"


df = pd.read_csv(file_path)

allBrightness = ['Peak 1 a','Peak 2 a','Peak 3 a','NIR a']
brightness = df[allBrightness ].to_numpy().flatten()
maxB =max(brightness )

condLen = 64

fname = "samped_DNA/" + "green_dis"
distr = np.load( fname + ".npy", allow_pickle=True )

DNASeqs = sample_loop(model_A, distr,
            cond_scales=[1.],  # list of cond scales - each sampled...
            num_samples=4,  # how many samples produced every time tested.....
            timesteps=96, flag=10000, foldproteins=True,
            calc_error=True, device=device)

fname = "samped_DNA/" + modelName  + "green_seqs.txt"
with open(fname , 'w') as f:
    for line in DNASeqs:
        f.write(f"{line}\n")
f.close()


fname = "samped_DNA/" + "red_dis"
distr = np.load( fname + ".npy", allow_pickle=True )


DNASeqs = sample_loop(model_A, distr,
            cond_scales=[1.],  # list of cond scales - each sampled...
            num_samples=4,  # how many samples produced every time tested.....
            timesteps=96, flag=10000, foldproteins=True,
            calc_error=True, device=device)

fname = "samped_DNA/" + modelName  + "red_seqs.txt"
with open(fname , 'w') as f:
    for line in DNASeqs:
        f.write(f"{line}\n")
f.close()


fname = "samped_DNA/" + "far_red_dis"
distr = np.load( fname + ".npy", allow_pickle=True )

DNASeqs = sample_loop(model_A, distr,
            cond_scales=[1.],  # list of cond scales - each sampled...
            num_samples=4,  # how many samples produced every time tested.....
            timesteps=96, flag=10000, foldproteins=True,
            calc_error=True, device=device)

fname = "samped_DNA/" + modelName  + "far_red_seqs.txt"
with open(fname , 'w') as f:
    for line in DNASeqs:
        f.write(f"{line}\n")
f.close()


fname = "samped_DNA/" + "nir_dis"
distr = np.load( fname + ".npy", allow_pickle=True )

DNASeqs = sample_loop(model_A, distr,
            cond_scales=[1.],  # list of cond scales - each sampled...
            num_samples=4,  # how many samples produced every time tested.....
            timesteps=96, flag=10000, foldproteins=True,
            calc_error=True, device=device)

fname = "samped_DNA/" + modelName  + "nir_seqs.txt"
with open(fname , 'w') as f:
    for line in DNASeqs:
        f.write(f"{line}\n")
f.close()


fname = "samped_DNA/" + "nir_2_dis"
distr = np.load( fname + ".npy", allow_pickle=True )


DNASeqs = sample_loop(model_A, distr,
            cond_scales=[1.],  # list of cond scales - each sampled...
            num_samples=4,  # how many samples produced every time tested.....
            timesteps=96, flag=10000, foldproteins=True,
            calc_error=True, device=device)

fname = "samped_DNA/" + modelName  + "nir_2_seqs.txt"
with open(fname , 'w') as f:
    for line in DNASeqs:
        f.write(f"{line}\n")
f.close()




# In[65]:


norm_flag = False
flag_ref = 40000


# In[66]:


#### Generate candidates

norm_flag = False
flag_ref = 40000
