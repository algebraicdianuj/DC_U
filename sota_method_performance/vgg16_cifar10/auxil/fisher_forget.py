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
from tqdm import tqdm




def hessian(dataset, model, device):
    model.eval()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    for p in model.parameters():
        p.grad_acc = torch.zeros_like(p.data)
        p.grad2_acc = torch.zeros_like(p.data)
    
    for data, orig_target in tqdm(train_loader, desc="Computing Hessian"):
        data, orig_target = data.to(device), orig_target.to(device)
        output = model(data)
        prob = F.softmax(output, dim=-1).data

        for y in range(output.shape[1]):
            target = torch.empty_like(orig_target).fill_(y)
            loss = loss_fn(output, target)
            model.zero_grad()
            loss.backward(retain_graph=True)
            for p in model.parameters():
                if p.requires_grad:
                    p.grad_acc += (orig_target == target).float() * p.grad.data
                    p.grad2_acc += prob[:, y] * p.grad.data.pow(2)
    
    for p in model.parameters():
        p.grad_acc /= len(train_loader)
        p.grad2_acc /= len(train_loader)

def get_mean_var(p, is_base_dist=False, alpha=3e-6, num_classes=10, num_to_forget=None, class_to_forget=None):
    var = copy.deepcopy(1. / (p.grad2_acc + 1e-8))
    var = var.clamp(max=1e3)
    if p.size(0) == num_classes:
        var = var.clamp(max=1e2)
    var = alpha * var
    
    if p.ndim > 1:
        var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
    if not is_base_dist:
        mu = copy.deepcopy(p.data.clone())
    else:
        mu = copy.deepcopy(p.data.clone())
    if p.size(0) == num_classes and num_to_forget is None:
        mu[class_to_forget] = 0
        var[class_to_forget] = 0.0001
    if p.size(0) == num_classes:
        var *= 10
    elif p.ndim == 1:
        var *= 10
    return mu, var

def kl_divergence_fisher(mu0, var0, mu1, var1):
    return ((mu1 - mu0).pow(2)/var0 + var1/var0 - torch.log(var1/var0) - 1).sum()

def check_parameter_updates(model_before, model_after):
    updated_parameters = []
    for (name_before, param_before), (name_after, param_after) in zip(model_before.named_parameters(), model_after.named_parameters()):
        if not torch.allclose(param_before.data, param_after.data):
            updated_parameters.append(name_before)
    return updated_parameters

def fisher_forgetting(model, retain_loader, forget_loader, device, alpha=1e-6):
    model_fisher = copy.deepcopy(model)
    model_base = copy.deepcopy(model)

    for p in model_fisher.parameters():
        p.data0 = copy.deepcopy(p.data.clone())

    hessian(retain_loader.dataset, model_fisher, device)
    hessian(forget_loader.dataset, model_base, device)

    # torch.manual_seed(seed)
    total_kl = 0
    for (k, p), (k0, p0) in zip(model_fisher.named_parameters(), model_base.named_parameters()):
        mu0, var0 = get_mean_var(p, False, alpha=alpha)
        mu1, var1 = get_mean_var(p0, True, alpha=alpha)
        kl = kl_divergence_fisher(mu0, var0, mu1, var1).item()
        total_kl += kl
        print(f"{k}: KL divergence = {kl:.1f}")

        p.data = p.data0 - var0.sqrt() * torch.empty_like(p.data0).normal_()

    print(f"Total KL divergence: {total_kl}")

    return model_fisher



