
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import numpy as np
import time
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
from utils.utils import *
from tqdm import tqdm
from copy import deepcopy



def hessian(dataset, model, device, num_classes):
    model.eval()
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    for p in model.parameters():
        p.grad2_acc = torch.zeros_like(p.data)
    
    for data, orig_target in tqdm(train_loader):
        data, orig_target = data.to(device), orig_target.to(device)
        output = model(data)
        prob = F.softmax(output, dim=-1).detach()

        for y in range(num_classes):
            target = torch.full_like(orig_target, y) 
            loss = loss_fn(output, target)
            model.zero_grad()
            loss.backward(retain_graph=True)
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad2_acc += prob[:, y].mean() * p.grad.data.pow(2)
    

    for p in model.parameters():
        p.grad2_acc /= len(train_loader)


def get_mean_var(p, is_base_dist=False, alpha=3e-6, num_classes=None, class_to_forget=None):

    var = deepcopy(1.0 / (p.grad2_acc + 1e-8))
    
    var = var.clamp(max=1e3)
    if num_classes is not None and p.size(0) == num_classes:
        var = var.clamp(max=1e2)
    
    var = alpha * var

    if p.ndim > 1:
        var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
    

    mu = deepcopy(p.data0)

    if class_to_forget is not None and num_classes is not None and p.size(0) == num_classes:
        mu[class_to_forget] = 0.0
        var[class_to_forget] = 1e-4  
    
    if num_classes is not None and p.size(0) == num_classes:
        var *= 10
    elif p.ndim == 1:
        var *= 10

    return mu, var


def fisher_forgetting(model, retain_loader, num_classes, device, 
                      class_to_forget=None, num_to_forget=None, alpha=1e-6):

    for p in model.parameters():
        p.data0 = deepcopy(p.data)

    hessian(retain_loader.dataset, model, device, num_classes)


    for p in model.parameters():
        mu, var = get_mean_var(
            p, 
            is_base_dist=False, 
            alpha=alpha, 
            num_classes=num_classes, 
            class_to_forget=class_to_forget
        )
        p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()

    return model