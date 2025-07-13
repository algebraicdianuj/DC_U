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




def retraining(model, criterion, device, learning_rate, momentum, weight_decay,warmup, retraining_epochs, retain_loader, decreasing_lr="50,75"):
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )

    lambda0 = lambda cur_iter: (cur_iter + 1) / warmup if cur_iter < warmup else (
        0.5 * (1.0 + np.cos(np.pi * ((cur_iter - warmup) / (retraining_epochs - warmup))))
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    
    model.train()
    for epoch in range(retraining_epochs):
        for batch in retain_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step() 


    for name, param in model.named_parameters():
        param.requires_grad = False

    return model




