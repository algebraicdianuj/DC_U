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
from auxil.retrain import *
from torch.nn import Conv2d, Linear


def FT_iter(unlearn_epochs, train_loader, model, criterion, optimizer, epoch, no_l1_epochs, alpha, device):

    model.train()

    for data in train_loader:
        image, target = data[0].to(device), data[1].to(device)

            
        if epoch < unlearn_epochs - no_l1_epochs:
            current_alpha = alpha * (1 - epoch / (unlearn_epochs - no_l1_epochs))
        else:
            current_alpha = 0

        output_clean = model(image)
        loss = criterion(output_clean, target)
        loss += current_alpha * l1_regularization(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def sparse_unlearning(retain_loader, model, lr, unlearning_epochs, no_l1_epochs=15, alpha=1e-4, device=torch.device('cpu')):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(unlearning_epochs): 
        FT_iter(unlearning_epochs, retain_loader, model, criterion, optimizer, epoch,  no_l1_epochs, alpha, device)

    for name, param in model.named_parameters():
        param.requires_grad = False

    return model



def sparse_unlearning_pruned(retain_loader, model, lr, unlearning_epochs, no_l1_epochs=15, alpha=1e-4, prune_ratio = 0.05, device=torch.device('cpu')):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(unlearning_epochs): 
        FT_iter(unlearning_epochs, retain_loader, model, criterion, optimizer, epoch,  no_l1_epochs, alpha, device)


    pruning_model(model, prune_ratio)

    for name, param in model.named_parameters():
        param.requires_grad = False

    return model


def synflow_importance_score(
    model,
    dataloader,
):
    @torch.no_grad()
    def linearize(model):
        signs = {}
        for name, param in model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(model, signs):
        # model.float()
        for name, param in model.state_dict().items():
            param.mul_(signs[name])

    model.eval()  # Crucial! BatchNorm will break the conservation laws for synaptic saliency
    model.zero_grad()
    score_dict = {}
    signs = linearize(model)

    (data, _) = next(iter(dataloader))
    input_dim = list(data[0, :].shape)
    input = torch.ones([1] + input_dim).to(next(model.parameters()).device)
    output = model(input)
    torch.sum(output).backward()

    for m in model.modules():
        if isinstance(m, (Conv2d, Linear)):  # Add Linear here
            if hasattr(m, "weight_orig"):
                score_dict[(m, "weight")] = (m.weight_orig.grad.data * m.weight_orig.data).abs()
            else:
                score_dict[(m, "weight")] = (m.weight.grad.data * m.weight.data).abs()
            if m.bias is not None:  # Optionally, handle biases too if you're pruning them
                if hasattr(m, "bias_orig"):
                    score_dict[(m, "bias")] = (m.bias_orig.grad.data * m.bias_orig.data).abs()
                else:
                    score_dict[(m, "bias")] = (m.bias.grad.data * m.bias.data).abs()
    model.zero_grad()
    nonlinearize(model, signs)
    return score_dict


def prune_and_retrain(model, criterion, device, learning_rate, retraining_epochs, retain_loader, prune_ratio=0.95):
    iteration_number = 100  # In SynFlow Paper, an iteration number of 100 performs well

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    each_ratio = 1 - (1 - prune_ratio) ** (1 / iteration_number)
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, Conv2d) or isinstance(module, Linear):  # include other types if necessary
            parameters_to_prune.append((module, 'weight'))
    
    for _ in range(iteration_number):
        score_dict = synflow_importance_score(model, retain_loader)
        importance_scores = {param: score_dict[(module, 'weight')] for module, param in parameters_to_prune if (module, 'weight') in score_dict}

        prune.global_unstructured(
            parameters=parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=each_ratio,
            importance_scores=importance_scores,
        )

    final_unlearned_model = retraining(model, criterion, device, learning_rate, retraining_epochs, retain_loader)
    return final_unlearned_model