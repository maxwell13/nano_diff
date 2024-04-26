

from nano_dif import ProteinDesigner_A
import torch
from nanoDataPrep  import  load_data_set_seq2seq

import  numpy as np

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
    DNASeqs = []
    for item in train_loader:

        X_train_batch = item[0].to(device)
        y_train_batch = item[1].to(device)

        GT = y_train_batch.cpu().detach()

        num_samples = min(num_samples, y_train_batch.shape[0])
        print(f"Producing {num_samples} samples...")

        charToInt = {"A": 1, "C": 2, "G": 3, "T": 4}
        intToChar = dict((v, k) for k, v in charToInt.items())

        train_unet_number = 1
        # max one hot enocded dna

        for iisample in range(len(cond_scales)):

            result = model.sample(X_train_batch, stop_at_unet_number=  train_unet_number,
                                  cond_scale=cond_scales[iisample])


            for ind  in range( result.shape[0]):
                samp = result[ind,:,:]
                sampResSeq = ""
                [_,totSize]= samp.shape
                for secS in range(0,totSize,4):
                    sec = samp[:,secS:secS+4]
                    pos = np.argmax(sec.numpy())
                    let =  intToChar[pos+1]
                    sampResSeq +=let
            DNASeqs.append( sampResSeq )

        if steps > num_samples:
            break


    print("done sampling ")

device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")

torch.rand(10).to(device)

print(" hopefully running on ")
print( device )

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


model_A = ProteinDesigner_A(timesteps=(96), dim=dim, pred_dim=pred_dim,
                            loss_type=0, elucidated=True,
                            padding_idx=0,
                            cond_dim=cond_dim,
                            text_embed_dim=cond_dim - embed_dim_position,
                            embed_dim_position=embed_dim_position,
                            max_seq_len=max_length,
                            max_text_len=max_text_len,
                            device=device,
                            ).to(device)



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
    batch_size_ = 16

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
