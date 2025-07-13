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
import torchvision.models as models
import timm

    
class vitl(nn.Module):
    def __init__(self, im_size, num_classes=10):
        super(vitl, self).__init__()
        # Load the pre-trained ViT model
        self.vit_model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
        
        # Modify the embedding layer to accept 32x32 images
        self.vit_model.patch_embed = timm.models.vision_transformer.PatchEmbed(
            img_size=im_size, patch_size=4, embed_dim=192)
        
        # Update position embeddings
        num_patches = self.vit_model.patch_embed.num_patches
        self.vit_model.pos_embed = nn.Parameter(self.vit_model.pos_embed.new_zeros(1, num_patches + 1, self.vit_model.embed_dim))
        
        # Modify the classifier to give output of size 10
        self.vit_model.head = nn.Linear(self.vit_model.head.in_features, num_classes)

    def forward(self, x):
        return self.vit_model(x)
    
    def feature(self, x):
        B = x.shape[0]
        x = self.vit_model.patch_embed(x)
        
        cls_tokens = self.vit_model.cls_token.expand(B, -1, -1)  # Stole cls_tokens implementation from forward
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.vit_model.pos_embed
        x = self.vit_model.pos_drop(x)
        
        for blk in self.vit_model.blocks:
            x = blk(x)
        
        x = self.vit_model.norm(x)
        return x[:, 0]  # Returning only the cls_token features




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
        self.patch_embed = original_model.vit_model.patch_embed  # Patch Embedding
        self.cls_token = original_model.vit_model.cls_token  # Classification Token
        self.pos_embed = original_model.vit_model.pos_embed  # Positional Embedding
        self.pos_drop = original_model.vit_model.pos_drop  # Position Dropout
        self.blocks = nn.ModuleList(original_model.vit_model.blocks[:2])  # First two transformer blocks

    def forward(self, x):
        B, _, _, _ = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (Batch_size, 1, embed_dim)
        x = self.patch_embed(x)
        x = torch.cat((cls_tokens, x), dim=1)  # (Batch_size, num_patches + 1, embed_dim)
        x = x + self.pos_embed  # Add positional embedding
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        return x

class Intermediate(nn.Module):
    def __init__(self, original_model):
        super(Intermediate, self).__init__()
        self.blocks = nn.ModuleList(original_model.vit_model.blocks[2:])  # Remaining transformer blocks
        self.norm = original_model.vit_model.norm  # Layer Normalization

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

class Final(nn.Module):
    def __init__(self, original_model):
        super(Final, self).__init__()

        self.fc_norm = original_model.vit_model.fc_norm
        self.head_drop = original_model.vit_model.head_drop  # Dropout layer
        self.head = original_model.vit_model.head  # Classification head
        
    def forward(self, x):
        x = self.fc_norm(x)
        x = self.head_drop(x)
        x = x[:, 0]
        return self.head(x)



