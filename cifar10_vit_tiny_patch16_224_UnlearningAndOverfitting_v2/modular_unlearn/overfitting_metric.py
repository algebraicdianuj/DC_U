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
from skimage.filters import threshold_otsu
from auxil.auxils import *



def overfitting_metric(model,criterion,input,label):
    model.train()
    output=model(input)
    loss = criterion(output, label)

    loss_grads = torch.autograd.grad(loss, model.parameters())
    loss_grads = list((_.detach().clone() for _ in loss_grads))

    total_sum = 0
    total_elements = 0
    for grad_tensor in loss_grads:
        squared_tensor = grad_tensor ** 2

        tensor_sum = torch.sum(squared_tensor)

        total_sum += tensor_sum.item()

        total_elements += squared_tensor.numel()

    mean_square_loss_grads = total_sum / total_elements

    overfitting_metric = torch.abs(loss-mean_square_loss_grads)

    return overfitting_metric


def get_overfitting_samples(model,criterion,training_images, training_labels, indices, device):

    images=training_images.to(device)
    labels= training_labels.to(device)

    measure=[]
    for i in range(len(images)):
        measure.append(overfitting_metric(model,criterion,images[i:i+1],labels[i:i+1]).cpu().detach().item())


    threshold = threshold_otsu(np.array(measure))

    print('------Stats of Otsu------')
    print('max measure: ', max(measure))
    print('min measure: ', min(measure))
    print('threshold: ', threshold)
    print('--------------------------')

    binary_measure = [0 if mv > threshold else 1 for mv in measure]

    # find indices of binary_measure that are 1
    indices_forget = [i for i, x in enumerate(binary_measure) if x == 1]
    indices_retain = [i for i, x in enumerate(binary_measure) if x == 0]

    super_forget_indices = indices[indices_forget]
    super_retain_indices = indices[indices_retain]
    
    forget_images, forget_labels= training_images[super_forget_indices], training_labels[super_forget_indices]
    retain_images, retain_labels= training_images[super_retain_indices], training_labels[super_retain_indices]

    return super_forget_indices, super_retain_indices, forget_images, forget_labels, retain_images, retain_labels



def measure_overfitting(model,criterion, dataloader,device):
    model.eval()
    training_images=[]
    training_labels=[]
    for images, labels in dataloader:
        training_images.append(images)
        training_labels.append(labels)
    training_images=torch.cat(training_images)
    training_labels=torch.cat(training_labels)

    images=training_images.to(device)
    labels= training_labels.to(device)

    measure=[]
    for i in range(len(images)):
        measure.append(overfitting_metric(model,criterion,images[i:i+1],labels[i:i+1]).cpu().detach().item())

    return np.mean(measure)







