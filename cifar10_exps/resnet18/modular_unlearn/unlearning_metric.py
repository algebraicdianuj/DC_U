
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



def get_samples_by_class(dataloader, target_class, lim=256):
    img_samples = []
    lab_samples = []
    for inputs, labels in dataloader:
        for i in range(len(labels)):
            if labels[i] == target_class:
                img_samples.append(inputs[i])
                lab_samples.append(labels[i])

    img_samples = torch.stack(img_samples)[:lim]
    lab_samples = torch.stack(lab_samples)[:lim]
    return img_samples, lab_samples




def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4:  # conv, out*in*h*w
        gwr = gwr.view(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.view(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.view(shape[0], shape[1] * shape[2])
        gws = gws.view(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:  # linear, out*in
        pass  # Do nothing
    elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.view(1, shape[0])
        gws = gws.view(1, shape[0])
        return torch.tensor(0.0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.mean(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))

    return dis_weight



def match_loss(gw_forget, gw_retain,device):  # authors proposed match_loss metric
    dis = torch.tensor(0.0, dtype=torch.float, device=device)

    for ig in range(len(gw_retain)):
        gwr = gw_retain[ig]
        gws = gw_forget[ig]
        dis += distance_wb(gwr, gws)

    return dis/len(gw_retain)



def measure_unlearning(unlearned_model,forget_loader,retain_loader,device):
    unlearned_model.to(device)
    for param in unlearned_model.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()

    # Find unique classes in forget_loader
    unique_classes = set()
    for _, labels in forget_loader:
        unique_classes.update(labels.numpy())

    unlearning_metric = 0.0
    for cl in unique_classes:
        # Get samples from forget_loader and retain_loader
        forget_samples, forget_labels = get_samples_by_class(forget_loader, cl)
        retain_samples, retain_labels = get_samples_by_class(retain_loader, cl)
        # Get predictions for forget_samples and retain_samples
        forget_preds = unlearned_model(forget_samples.to(device))
        retain_preds = unlearned_model(retain_samples.to(device))

        # Calculate loss for forget_samples and retain_samples
        forget_loss = criterion(forget_preds, forget_labels.to(device))
        retain_loss = criterion(retain_preds, retain_labels.to(device))

        # Calculate gradient for forget_samples and retain_samples
        forget_grads = torch.autograd.grad(forget_loss, unlearned_model.parameters())
        forget_grads = list((_.detach().clone() for _ in forget_grads))
        retain_grads = torch.autograd.grad(retain_loss, unlearned_model.parameters())
        retain_grads = list((_.detach().clone() for _ in retain_grads))

        # Calculate match_loss for forget_samples and retain_samples
        unlearning_metric += match_loss(forget_grads, retain_grads,device).item()

    return unlearning_metric/len(unique_classes)*100

    

    



