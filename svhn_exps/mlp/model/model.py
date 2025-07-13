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



# Define your network architecture (MLP)
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size*4)
        self.fc2 = nn.Linear(hidden_size*4, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        return y


    def feature(self,x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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
    def __init__(self, input_size, hidden_size):
        super(Beginning, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size*4)

    def forward(self,x):
        x=x.view(x.size(0), -1)
        x=F.relu(self.fc1(x))
        return x

class Intermediate(nn.Module):
    def __init__(self,hidden_size):
        super(Intermediate, self).__init__()
        self.fc2 = nn.Linear(hidden_size*4, hidden_size)

    def forward(self,x):
        x=F.relu(self.fc2(x))
        return x


class Final(nn.Module):
    def __init__(self,hidden_size, num_classes):
        super(Final, self).__init__()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self,x):
        x=F.relu(self.fc3(x))
        return x
