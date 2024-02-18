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
from opacus import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from auxil.auxils import *


def DP_Adam(net,train_loader,max_grad_norm, epsilon, delta, multiplier, train_lr, train_epochs, device):
    for param in net.parameters():
        param.requires_grad = True
    net_copy = copy.deepcopy(net)
    optimizer = torch.optim.Adam(net.parameters(),lr=train_lr)
    # privacy_engine = PrivacyEngine(
    #     net,
    #     batch_size=train_loader.batch_size,
    #     sample_size=len(train_loader.dataset),
    #     alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
    #     noise_multiplier=multiplier,
    #     max_grad_norm=max_grad_norm,
    # )
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=net,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=multiplier,  # Set your desired noise multiplier here
        max_grad_norm=max_grad_norm,  # Set your desired max gradient norm here
    )

    # model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    #     module=net,
    #     optimizer=optimizer,
    #     data_loader=train_loader,
    #     epochs=2,
    #     target_epsilon=epsilon,
    #     target_delta=delta,
    #     max_grad_norm=max_grad_norm,
    # )

    for epoch in range(train_epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

    
    # copy the parameters from the trained model to the net_copy
    for param, param_copy in zip(net.parameters(), net_copy.parameters()):
        param_copy.data = param.data.clone()

    for param in net_copy.parameters():
        param.requires_grad = False

    return net_copy