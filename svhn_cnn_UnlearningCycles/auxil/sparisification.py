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
from auxil.auxils import *



def FT_iter(unlearn_epochs, train_loader, model, criterion, optimizer, epoch, no_l1_epochs, alpha, device):

    model.train()

    for data in train_loader:
        image, target = data[0].to(device), data[1].to(device)

            
        if epoch < unlearn_epochs - no_l1_epochs:
            current_alpha = alpha * (1 - epoch / (unlearn_epochs - no_l1_epochs))
        else:
            current_alpha = 0

        output_clean = model(image)
        loss = criterion(output_clean, target)
        loss += current_alpha * l1_regularization(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def sparse_unlearning(retain_loader, model, lr, unlearning_epochs, no_l1_epochs=15, alpha=1e-4, device=torch.device('cpu')):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(unlearning_epochs): 
        FT_iter(unlearning_epochs, retain_loader, model, criterion, optimizer, epoch,  no_l1_epochs, alpha, device)

    for name, param in model.named_parameters():
        param.requires_grad = False

    return model



def sparse_unlearning_pruned(retain_loader, model, lr, unlearning_epochs, no_l1_epochs=15, alpha=1e-4, prune_ratio = 0.05, device=torch.device('cpu')):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(unlearning_epochs): 
        FT_iter(unlearning_epochs, retain_loader, model, criterion, optimizer, epoch,  no_l1_epochs, alpha, device)


    pruning_model(model, prune_ratio)

    for name, param in model.named_parameters():
        param.requires_grad = False

    return model