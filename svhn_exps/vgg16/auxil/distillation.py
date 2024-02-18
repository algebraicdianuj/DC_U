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




def distillation_unlearning(retain_loader, train_lr, model_student, model_teacher, epochs, device, alpha=1, gamma=1, kd_T=4.0):
    optimizer_s = torch.optim.Adam(model_student.parameters(), lr=train_lr)
    criterion_class = nn.CrossEntropyLoss()
    criterion_divergence = DistillKL(kd_T)

    for epoch in range(epochs):

        for data in retain_loader:
            in_data,target=data
            in_data=in_data.to(device)
            target=target.to(device)

            logit_student=model_student(in_data)

            with torch.no_grad():
                logit_teacher=model_teacher(in_data)   # because we want to optimize student model, not teacher model


            loss_class=criterion_class(logit_student,target)
            # for reference: this divergence calculate how much behavior of student model is deflected from teacher model on the given dataset (which can be either retain or forget)
            loss_divergence=criterion_divergence(logit_student,logit_teacher)   #represent d(xr; wu) in the paper in case of minimization, and d(xf; wu) in case of maximization
            # certainly minimization will be performed on the retained dataset, and maximization will be performed on the forgotten dataset
    
            loss=alpha*loss_class+gamma*loss_divergence


            # performing backward propogation of losses
            optimizer_s.zero_grad()
            loss.backward()
            optimizer_s.step()   # optimize student model


    for name, param in model_student.named_parameters():
        param.requires_grad = False


    return model_student