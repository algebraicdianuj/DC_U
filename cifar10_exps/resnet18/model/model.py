import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import numpy as np
import time
import copy
from torch.utils.data import TensorDataset
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import rotate as scipyrotate
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from sklearn import linear_model, model_selection
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from sklearn import linear_model, model_selection
import torchvision.models as models
from sklearn.cluster import KMeans
from collections import OrderedDict


class Resnet18(nn.Module):
    def __init__(self, original_model):
        super(Resnet18, self).__init__()
        self.original_model = original_model
        
    def forward(self, x):
        return self.original_model(x)
        
    def feature(self, x):
        for name, layer in self.original_model.named_children():
            x = layer(x)
            if name == 'avgpool':
                x = x.view(x.size(0), -1)
                return x

            # if name == 'layer1':
            #     x = x.view(x.size(0), -1)
            #     return x



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class InverterResnet18(nn.Module):
    def __init__(self, output_size):
        super(InverterResnet18, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, output_size[0]*output_size[1]*output_size[2])
        self.output_size = output_size
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        strides = [stride] + [1]*(num_blocks-1)
        for stride in strides:
            layers.append(ResidualBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = out.view(out.size(0), self.output_size[0],self.output_size[1],self.output_size[2])  # Reshape to match CIFAR-10 image shape
        return out

    


class Beginning(nn.Module):
    def __init__(self, original_model):
        super(Beginning, self).__init__()
        self.features = nn.Sequential(
            original_model.original_model.conv1,
            original_model.original_model.bn1,
            original_model.original_model.relu,
            original_model.original_model.maxpool,
            original_model.original_model.layer1,
            original_model.original_model.layer2,
        )
        
    def forward(self, x):
        return self.features(x)


class Intermediate(nn.Module):
    def __init__(self, original_model):
        super(Intermediate, self).__init__()
        self.features = original_model.original_model.layer3
        
    def forward(self, x):
        return self.features(x)


class Final(nn.Module):
    def __init__(self, original_model):
        super(Final, self).__init__()
        self.features = nn.Sequential(
            original_model.original_model.layer4,
            original_model.original_model.avgpool,
        )
        self.fc = original_model.original_model.fc
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

