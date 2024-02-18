
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

def add_noise(combined_model, forget_loader, criterion, threshold, lambd, device):

    # Calculate the Fisher Information Matrix for each parameter in base_model
    fisher_information = {}
    for name, param in combined_model.named_parameters():
        fisher_information[name] = torch.zeros_like(param).to(device)

    # Assume we use a single datapoint to calculate the Fisher Information
    # Usually, you would use a dataset or a subset
    for i, (inputs, labels) in enumerate(forget_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        combined_model.zero_grad()
        outputs = combined_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in combined_model.named_parameters():
            fisher_information[name] += param.grad ** 2 / len(forget_loader)


    # Save the optimal parameters for distill_loader
    optimal_params = {}
    for name, param in combined_model.named_parameters():
        optimal_params[name] = param.clone()


    # Normalize and binarize the Fisher Information

    for name, param in fisher_information.items():
        # Normalizing by dividing each entry by the maximum value
        param /= torch.max(param)

        # Binarizing by applying a threshold
        param[param < threshold] = 0
        param[param >= threshold] = 1


    for name, param in combined_model.named_parameters():
        noise = torch.randn_like(param)
        noise *= lambd*fisher_information[name]
        param.data += noise


    return combined_model




def modular_unlearning(combined_model, optim_model, criterion, device, beggining_epochs, intermediate_epochs, final_epochs, overture_epochs, final_thr, img_syn_loader, reduced_retain_loader):
    for main_ep in range(overture_epochs):

        for param in list(combined_model.databank.beggining.parameters()):
            param.requires_grad = True

        for param in list(combined_model.databank.intermediate.parameters()):
            param.requires_grad = False

        for param in list(combined_model.final.parameters()):
            param.requires_grad = False


        for _ in range(beggining_epochs):
            for batch in reduced_retain_loader:
                img,lab=batch
                img,lab=img.to(device), lab.to(device)
                output=combined_model(img)
                loss=criterion(output, lab)

                optim_model.zero_grad()
                loss.backward()

                optim_model.step()


        for param in list(combined_model.parameters()):
            param.requires_grad = False

        for param in list(combined_model.final.parameters()):
            param.requires_grad = True


        if main_ep<overture_epochs-final_thr:

            for epi in range(final_epochs):
                distill_loss=0.0
                for batch in img_syn_loader:
                    img,lab=batch
                    img,lab=img.to(device), lab.to(device)
                    output=combined_model(img)
                    loss=criterion(output, lab)
                    distill_loss+=loss
                distill_loss/=len(img_syn_loader)

                lhs_loss=distill_loss

                optim_model.zero_grad()
                lhs_loss.backward()
                optim_model.step()

        for param in list(combined_model.parameters()):
            param.requires_grad = False


    #---------------------just training the intermediate--------------------------


    for param in list(combined_model.databank.intermediate.parameters()):
        param.requires_grad = True


    for second_ep in range(intermediate_epochs):
        for batch in reduced_retain_loader:
            img,lab=batch
            img,lab=img.to(device), lab.to(device)
            output=combined_model(img)
            loss=criterion(output, lab)
            optim_model.zero_grad()
            loss.backward()
            optim_model.step()

    for param in list(combined_model.parameters()):
        param.requires_grad = False


    return combined_model