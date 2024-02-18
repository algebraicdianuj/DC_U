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
import math
from fastDP import PrivacyEngine

# def calculate_epsilon(epochs, noise_multiplier, clipping_norm, batch_size, dataset_size, delta):
#     sampling_probability = batch_size / dataset_size
#     epsilon = (epochs * clipping_norm) / (noise_multiplier * sampling_probability) * math.sqrt(2 * math.log(1/delta))
#     return epsilon

def calculate_epsilon(noise_multiplier, l2_norm_clip, sigma, steps, delta):
    epsilon_1 = (noise_multiplier * l2_norm_clip) / sigma
    epsilon = epsilon_1 * math.sqrt(2 * math.log(1/delta) * steps)
    return epsilon

class DPsgd(torch.optim.SGD):
    def __init__(self, params, lr, noise_multiplier, l2_norm_clip):
        super().__init__(params, lr)
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip

    def step(self):
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad

                    # Clip the gradients
                    total_norm = grad.norm(2)
                    total_norm = total_norm.clamp(min=1e-6)
                    clip_coef = self.l2_norm_clip / total_norm
                    if clip_coef < 1:
                        grad.mul_(clip_coef)

                    # Add noise
                    noise = torch.randn_like(grad) * self.noise_multiplier
                    grad.add_(noise)

        super().step()




def DP_ADAM(net,train_loader,max_grad_norm, epsilon, delta, multiplier, train_lr, train_epochs, device):
    net=net.to(device)
    for param in net.parameters():
        param.requires_grad = True
    net_copy = copy.deepcopy(net)
    # optimizer = DPsgd(net.parameters(), lr=train_lr, noise_multiplier=multiplier, l2_norm_clip=max_grad_norm)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    privacy_engine = PrivacyEngine( net,
                                    batch_size=256,
                                    sample_size=50000,
                                    epochs=1,
                                    target_epsilon=10.0,
                                    clipping_fn='automatic',
                                    clipping_mode='MixOpt',
                                    origin_params=None,
                                    clipping_style='all-layer',)
    privacy_engine.attach(optimizer)
  


    
    # epsiloner=calculate_epsilon(multiplier, max_grad_norm, 1, train_epochs, delta)
    # # save epsiloner to file
    # with open('epsilon_naive.txt', 'w') as f:
    #     f.write(str(epsiloner))

    # print(f"(ε = {epsiloner:.2f}, δ = {delta})")


    for epoch in range(train_epochs):

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

        if (epoch+1)%5==0:
            print("Epoch: ", epoch)
            print(loss.item())
            train_acc = test(net, train_loader, device)
            print(f"Train Accuracy: {train_acc:.2f}")
            print("-------------------------------")


    
    # # copy the parameters from the trained model to the net_copy
    # for param, param_copy in zip(net.parameters(), net_copy.parameters()):
    #     param_copy.data = param.data.clone()

    # for param in net_copy.parameters():
    #     param.requires_grad = False

    return net