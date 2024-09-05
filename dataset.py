import numpy as np
import torch
import os

import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
import torchvision.transforms.functional as F

from tqdm.auto import tqdm
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from matplotlib import pyplot as plt
from PIL import Image

# %%
def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# %%
# You can change the size here to determine the targeted size like 64
target_size = 64

train_tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(target_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

test_tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop((target_size, target_size)),
    transforms.ToTensor(),
])


# %%
class CosDataset(Dataset):

    def __init__(self, imgs, labels, tfm=test_tfm, mode="cosmo"):
        super(CosDataset, self).__init__()
        # here I choose the unit as log10
        self.img_list = imgs
        if mode == "cosmo":
            self.label_list = labels[:,0:2]
        elif mode == "all":
            self.label_list = labels
        else:
            raise ValueError("Unsupported mode")
        self.transform = tfm
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        im = self.img_list[idx]
        label = self.label_list[idx]
        #im = Image.fromarray(np.float32(im), mode='L')
        im = self.transform(im)
        return im, label