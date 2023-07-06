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
from neural_vgg import DeviceDataLoader, evaluate

from torch.utils.data.dataloader import DataLoader

from torchvision.utils import make_grid

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


from ImageClassificationBase import ImageClassificationBase
from collections import OrderedDict




# class VGG16(nn.Module):
#     def __init__(self, num_classes=10):
#         super(VGG16, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU())
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(), 
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU())
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU())
#         self.layer6 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU())
#         self.layer7 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer8 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer9 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer10 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer11 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer12 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer13 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.fc = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(7*7*512, 4096),
#             nn.ReLU())
#         self.fc1 = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU())
#         self.fc2= nn.Sequential(
#             nn.Linear(4096, num_classes))
        
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.layer5(out)
#         out = self.layer6(out)
#         out = self.layer7(out)
#         out = self.layer8(out)
#         out = self.layer9(out)
#         out = self.layer10(out)
#         out = self.layer11(out)
#         out = self.layer12(out)
#         out = self.layer13(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.fc(out)
#         out = self.fc1(out)
#         out = self.fc2(out)
#         return out
    
# # def temp():
#     # # load pretrained weights
#     # model_vgg19 = vgg19(pretrained=True)
#     # sd_vgg19 = model_vgg19.state_dict()
        
#     # # init custom model (feature layers exactly like in vgg19)
#     # model = CustomNet()
#     # model_dict = model.state_dict()
        
#     # # rewrite to pretrained weights
#     # for key, val1, val2 in zip_dicts(sd_vgg19, model_dict):
#     #     # delete this condition if you want to rewrite classifier layers
#     #     if key.split('.')[0] != 'classifier':
#     #         model_dict[key] = sd_vgg19[key]
        
#     # # include pretrained weights to custom model
#     # model = CustomNet()
#     # model.load_state_dict(model_dict)

# # def zip_dicts(*dcts):
# #     """
# #     Helper function for zip dicts like zip(list1, list2) in python
# #     """
# #     if not dcts:
# #        return
# #     for i in set(dcts[0]).intersection(*dcts[1:]):
# #         yield (i,) + tuple(d[i] for d in dcts)
    
# if __name__ == '__main__':
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#     data_dir = './data/awa2'
#     test_dataset = ImageFolder(
#     data_dir+'/test', 
#     transform=transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ]))
#     device = torch.device("cuda")
#     test_loader = DeviceDataLoader(DataLoader(test_dataset, 32), device)


    
#     model = VGG16(5)
#     model.to(device)
#     model.load_state_dict(torch.load('vgg16-cnn.pth'), strict=False)
#     print(model)
#     print(evaluate(model, test_loader))
    
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
        

def get_vgg(num_classes, pretrained ,freeze=True):
    vgg = VGG()
    temp = torchvision.models.vgg16(pretrained=True)

    if pretrained == True:
        vgg.load_state_dict(temp.state_dict())
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
    model = get_vgg(pretrained=True,num_classes=5,freeze=True)
    print(model)
 
