
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



def get_samples_by_class(dataloader, target_class, num_samp=32):
    img_samples = []
    lab_samples = []
    for inputs, labels in dataloader:
        for i in range(len(labels)):
            if labels[i] == target_class:
                img_samples.append(inputs[i])
                lab_samples.append(labels[i])

    indices = list(range(len(img_samples)))
    # random.shuffle(indices)
    sliced_indices = indices[:num_samp]
    img_samples = torch.stack(img_samples)[sliced_indices]
    lab_samples = torch.stack(lab_samples)[sliced_indices]
    return img_samples, lab_samples


def cosine_similarity_between_grads(grads1, grads2):
    """
    Compute the cosine similarity between two lists of gradient tensors.
    """
    # Flatten the gradients tensors to 1D tensors
    grads1_flatten = torch.cat([g.view(-1) for g in grads1])
    grads2_flatten = torch.cat([g.view(-1) for g in grads2])
    
    # Compute the cosine similarity between the two flattened tensors
    cos_sim = F.cosine_similarity(grads1_flatten.unsqueeze(0), grads2_flatten.unsqueeze(0))
    
    return cos_sim.item()



def measure_unlearning(unlearned_model,forget_loader,retain_loader,device):
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


        unlearned_model.zero_grad()
        forget_preds = unlearned_model(forget_samples.to(device))
        forget_loss = criterion(forget_preds, forget_labels.to(device))
        # Calculate gradient for forget_samples and retain_samples
        forget_grads = torch.autograd.grad(forget_loss, unlearned_model.parameters())
        forget_grads = list((_.detach().clone() for _ in forget_grads))


        unlearned_model.zero_grad()
        retain_preds = unlearned_model(retain_samples.to(device))
        retain_loss = criterion(retain_preds, retain_labels.to(device))
        retain_grads = torch.autograd.grad(retain_loss, unlearned_model.parameters())
        retain_grads = list((_.detach().clone() for _ in retain_grads))

        # Calculate match_loss for forget_samples and retain_samples
        unlearning_metric += cosine_similarity_between_grads(forget_grads, retain_grads)

    return (1-unlearning_metric/len(unique_classes))*100

    

    



