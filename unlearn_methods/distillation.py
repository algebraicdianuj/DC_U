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
from utils.utils import *

def distillation_unlearning(retain_loader, 
                            train_lr, 
                            momentum,
                            weight_decay,
                            model_student, 
                            model_teacher, 
                            epochs, 
                            device, 
                            alpha=1, 
                            gamma=1, 
                            kd_T=4.0, 
                            decreasing_lr="50,75"):

    optimizer_s = torch.optim.SGD(
        model_student.parameters(),
        train_lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # Parse the LR schedule
    milestones = list(map(int, decreasing_lr.split(",")))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_s, milestones=milestones, gamma=0.1)

    criterion_class = nn.CrossEntropyLoss()
    criterion_divergence = DistillKL(kd_T)

    model_student.train()
    model_teacher.eval()

    for epoch in range(epochs):
        for in_data, target in retain_loader:
            in_data, target = in_data.to(device), target.to(device)

            logit_student = model_student(in_data)

            with torch.no_grad():
                logit_teacher = model_teacher(in_data)

            loss_class = criterion_class(logit_student, target)
            loss_divergence = criterion_divergence(logit_student, logit_teacher)

            loss = alpha * loss_class + gamma * loss_divergence

            optimizer_s.zero_grad()
            loss.backward()
            optimizer_s.step()

        scheduler.step()  

    for param in model_student.parameters():
        param.requires_grad = False

    return model_student
