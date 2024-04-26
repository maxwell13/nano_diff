import os,sys
import math
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #turn off CUDA if needed
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # select which GPU device is to be used

import shutil

from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import seaborn as sns

import torchvision

import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from functools import partial, wraps

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB import PDBList

from utils import once



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler

import torch
#from imagen_pytorch import Unet, Imagen

class RegressionDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)





device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
num_of_gpus = torch.cuda.device_count()

ynormfac=22.
batch_size_=512
max_length = 64
number = 99999999999999999
min_length=0
train_loader, train_loader_noshuffle, test_loader,tokenizer_y \
        = load_data_set_seq2seq (file_path='PROTEIN_Mar18_2022_SECSTR_ALL.csv',
                   min_length=0, max_length=max_length, batch_size_=batch_size_, output_dim=3,
                  maxdata=number,   split=0.1,)



import math
import copy
from random import random
from typing import List, Union
from tqdm.auto import tqdm
from functools import partial, wraps
from contextlib import contextmanager, nullcontext
from collections import namedtuple
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch import nn, einsum
from torch.cuda.amp import autocast
from torch.special import expm1
import torchvision.transforms as T

import kornia.augmentation as K

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from einops_exts import rearrange_many, repeat_many, check_shape
from einops_exts.torch import EinopsToAndFrom

# helper functions



print_once = once(print)

