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


class CNN(nn.Module):
    def __init__(self, channel, im_size, num_classes):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(channel, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # Calculating the size of the input for the first fully connected layer
        fc1_input_size = 1024
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc1_input_size, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flattening the output
        x = x.view(x.size(0), -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def feature(self,x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flattening the output
        x = x.view(x.size(0), -1)

        return x
        




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
    def __init__(self, cnn):
        super(Beginning, self).__init__()
     
        self.conv1 = cnn.conv1
        self.bn1 = cnn.bn1
        self.conv2 = cnn.conv2
        self.bn2 = cnn.bn2
        self.pool = cnn.pool

    def forward(self,x):

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        return x

class Intermediate(nn.Module):
    def __init__(self,cnn):
        super(Intermediate, self).__init__()
        self.conv3 = cnn.conv3
        self.bn3 = cnn.bn3
        self.conv4 = cnn.conv4
        self.bn4 = cnn.bn4
        # Pooling layer
        self.pool = cnn.pool
        

    def forward(self,x):
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flattening the output
        x = x.view(x.size(0), -1)
        return x


class Final(nn.Module):
    def __init__(self,cnn):
        super(Final, self).__init__()

        # Fully connected layers
        self.fc1 = cnn.fc1
        self.fc2 = cnn.fc2
        self.dropout = cnn.dropout

    def forward(self,x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

