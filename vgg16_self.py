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
# from neural_vgg import DeviceDataLoader, evaluate

from torch.utils.data.dataloader import DataLoader

from torchvision.utils import make_grid

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


from ImageClassificationBase import ImageClassificationBase
from collections import OrderedDict

    
class VGG(nn.Module):
    """
    Standard PyTorch implementation of VGG. Pretrained imagenet model is used.
    """
    def __init__(self):
        super().__init__()
    
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            
            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1000)
        )

        # We need these for MaxUnpool operation
        self.conv_layer_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        self.feature_maps = OrderedDict()
        self.pool_locs = OrderedDict()
        
    def forward(self, x):
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
            else:
                x = layer(x)
        
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
        

def get_vgg(num_classes, pretrained, device, freeze=True, load_from_path = False):
    vgg = VGG()
    vgg.to(device=device)
    temp = torchvision.models.vgg16(pretrained=True)

    if pretrained == True:
        vgg.load_state_dict(temp.state_dict())
    elif load_from_path == True:
        vgg.load_state_dict(torch.load('vgg16-cnn.pth'))
    
    if num_classes != 1000:
        n_inputs = vgg.classifier[6].in_features
        vgg.classifier[6] = nn.Linear(n_inputs, num_classes)
    if freeze == True:
        for name, param in vgg.named_parameters():
                if 'classifier.3' in name or 'classifier.6' in name:
                    continue
                else:
                    param.requires_grad = False

    total_params = sum(p.numel() for p in vgg.parameters())
    print(f'[VGG16] {total_params} total parameters.')
    total_trainable_params = sum(
    p.numel() for p in vgg.parameters() if p.requires_grad)
    print(f'[VGG16] {total_trainable_params} training parameters.')
    
    last_two = 0
    for name, param in vgg.named_parameters():
        if 'classifier.3' in name or 'classifier.6' in name:
            last_two += param.numel()
    print(f'[VGG16] {last_two} last two layers parameters.')

    return vgg

if __name__ == '__main__':
    model = get_vgg(pretrained=True,num_classes=5,freeze=True, device=torch.device('cuda'))
    print(model)
 
