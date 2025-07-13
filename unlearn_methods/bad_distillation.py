
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

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x = self.forget_data[index][0]
            y = 1
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = 0
        return x, y

def training_step(model, batch, device):
    images, labels, clabels = batch
    images, clabels = images.to(device), clabels.to(device)
    out = model(images)
    loss = F.cross_entropy(out, clabels)
    return loss

def UnlearnerLoss(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
    labels = torch.unsqueeze(labels, dim=1)
    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)
    overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out, reduction='batchmean') * (KL_temperature ** 2)

def unlearning_step(model, unlearning_teacher, full_trained_teacher, unlearn_data_loader, optimizer, 
                    device, KL_temperature):
    model.train()
    for batch in unlearn_data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)
        output = model(x)
        optimizer.zero_grad()
        loss = UnlearnerLoss(output, y, full_teacher_logits, unlearn_teacher_logits, KL_temperature)
        loss.backward()
        optimizer.step()



def blindspot_unlearner(model, unlearning_teacher, full_trained_teacher, retain_data, forget_data, epochs=10,
                        lr=0.01, momentum = 0.9, weight_decay = 5e-4, batch_size=256, device=torch.device('cpu'), KL_temperature=2.0,
                        decreasing_lr="50,75"):
    unlearning_data = UnLearningData(forget_data=forget_data, retain_data=retain_data)
    unlearning_loader = DataLoader(unlearning_data, batch_size=batch_size, shuffle=True)

    unlearning_teacher.eval()
    full_trained_teacher.eval()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    milestones = list(map(int, decreasing_lr.split(",")))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    for epoch in range(epochs):
        unlearning_step(model, unlearning_teacher, full_trained_teacher, unlearning_loader,
                        optimizer, device, KL_temperature)
        scheduler.step()

    return model
