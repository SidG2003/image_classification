import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline

from torch.utils.data.dataloader import DataLoader

from torchvision.utils import make_grid

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from ImageClassificationBase import ImageClassificationBase



class Cifar10CnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))
        
    def forward(self, xb):
        return self.network(xb)
    

def get_model(model_name, num_classes,pretrained = True):
    if model_name == 'cifar':
        return Cifar10CnnModel()
    
    elif model_name == 'vgg16':
        if pretrained == True:
            model = models.vgg16(pretrained=True)
            # Freeze early layers
            for param in model.parameters():
                param.requires_grad = False

        elif pretrained == False:
            model = models.vgg16(pretrained=False)
        
        n_inputs = model.classifier[6].in_features

        # Add on classifier
        # model.classifier[6] = nn.Sequential(
        #     nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.4),
        #     nn.Linear(256, num_classes), nn.LogSoftmax(dim=1)
        #     )
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
            )

        total_params = sum(p.numel() for p in model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')
        return model


if __name__ == "__main__":
    m = get_model('vgg16',10)
    print("--------------------------------------------")
    print("--------------------------------------------")
    print("--------------------------------------------")
    print("--------------------------------------------")
    

    print(m)



