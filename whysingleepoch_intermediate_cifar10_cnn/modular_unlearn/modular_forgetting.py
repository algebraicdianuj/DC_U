
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




def modular_unlearning(combined_model, optim_model, lr_overture, lr_intermediate, criterion, device, beggining_epochs, intermediate_epochs, final_epochs, overture_epochs, final_thr, img_syn_loader, reduced_retain_loader):
    optim_model.param_groups[0]['lr']=lr_overture


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

    optim_model.param_groups[0]['lr']=lr_intermediate

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



