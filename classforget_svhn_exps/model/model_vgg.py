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



class Vgg16(nn.Module):
    def __init__(self, vgg16):
        super(Vgg16, self).__init__()

        # Use the features from VGG16 (you can also modify this if needed)
        self.features = vgg16.features
        self.classifier = vgg16.classifier
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten the tensor
        x = self.classifier(x)
        return x
    
    # Method to get features
    def feature(self, x):
        x = self.features(x)
        x= x.view(x.size(0), -1) # Flatten the tensor
        return x
    


def modify_vgg16(channel, im_size, num_classes):
    vgg16 = models.vgg16(pretrained=False)
    
    # Modify the first convolutional layer to accept custom input shape
    vgg16.features[0] = nn.Conv2d(channel, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    # Create a dummy input with the shape of (1, channel, im_size, im_size)
    dummy_input = torch.randn(1, channel, im_size, im_size)
    
    # Pass the dummy input through the features part of the model
    features_output = vgg16.features(dummy_input)
    
    # Flatten the output from the features part of the model
    features_output_flatten = features_output.view(features_output.size(0), -1)
    
    # Determine the classifier input size based on the flattened output size
    classifier_input_size = 512
    
    # Adjusting the input features of the classifier
    vgg16.classifier[0] = nn.Linear(classifier_input_size, 4096)
    
    # Adjust the final classifier layer to have the desired number of classes as output
    vgg16.classifier[6] = nn.Linear(4096, num_classes)
    
    return vgg16




class ConvClassifier(nn.Module):
    def __init__(self, input_channels, im_size, num_classes):
        super(ConvClassifier, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate the size of the output feature maps
        # The -2 is to account for the kernel size and padding
        conv_output_size = im_size // 8
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * conv_output_size * conv_output_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


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
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = out.view(out.size(0), self.output_size[0],self.output_size[1],self.output_size[2])  # Reshape to match CIFAR-10 image shape
        return out


# Define your network architecture (MLP)
class InvertedMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(InvertedMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size*4)
        self.fc3 = nn.Linear(hidden_size*4, hidden_size*16)
        self.fc4 = nn.Linear(hidden_size*16, hidden_size*32)
        self.fc5 = nn.Linear(hidden_size*32, output_size[0]*output_size[1]*output_size[2])
        self.output_size = output_size


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x= F.relu(self.fc3(x))
        x= F.relu(self.fc4(x))
        y= self.fc5(x)
        y= y.reshape(y.shape[0],self.output_size[0],self.output_size[1],self.output_size[2])
        return y




class Beginning(nn.Module):
    def __init__(self, original_model):
        super(Beginning, self).__init__()
        self.features = nn.Sequential(*list(original_model.features.children())[:10])  # Adjust the slicing
        
    def forward(self, x):
        return self.features(x)


class Intermediate(nn.Module):
    def __init__(self, original_model):
        super(Intermediate, self).__init__()
        self.features = nn.Sequential(*list(original_model.features.children())[10:20])  # Adjust the slicing
        
    def forward(self, x):
        return self.features(x)


class Final(nn.Module):
    def __init__(self, original_model):
        super(Final, self).__init__()
        self.features = nn.Sequential(*list(original_model.features.children())[20:])  # Adjust the slicing
        self.classifier = original_model.classifier
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)





