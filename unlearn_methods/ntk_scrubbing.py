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
from tqdm import tqdm
import copy
from collections import OrderedDict




def ntk_scrubbing(model, retain_loader, forget_loader, device='cuda', num_classes=10, weight_decay=1e-1, beta=1e2):
    # Helper functions
    def vectorize_params(model):
        params = []
        for p in model.parameters():
            params.append(p.data.view(-1))
        return torch.cat(params).cpu()

    def delta_w_utils(model, dataloader, num_classes):
        model.eval()
        dataloader = DataLoader(dataloader.dataset, batch_size=1, shuffle=False)
        G_list = []
        f0_minus_y = []

        for idx, (input, target) in enumerate(tqdm(dataloader, leave=False)):
            input, target = input.to(device), target.to(device)
            output = model(input)

            for cls in range(num_classes):
                grads = torch.autograd.grad(output[0, cls], model.parameters(), retain_graph=True)
                grads = torch.cat([g.view(-1) for g in grads]).cpu()
                G_list.append(grads)

            p = F.softmax(output, dim=1).cpu().detach().transpose(0, 1)
            p[target.item()] -= 1
            f0_minus_y.append(p)

        return torch.stack(G_list).transpose(0, 1).to(device), torch.cat(f0_minus_y).to(device)

    def compute_ntk_matrices(model, retain_loader, forget_loader, num_classes, weight_decay):
        G_r, f0_minus_y_r = delta_w_utils(model, retain_loader, num_classes)
        G_f, f0_minus_y_f = delta_w_utils(model, forget_loader, num_classes)

        G = torch.cat([G_r, G_f], dim=1)
        f0_minus_y = torch.cat([f0_minus_y_r, f0_minus_y_f])

        num_total = len(retain_loader.dataset) + len(forget_loader.dataset)
        theta = G.transpose(0, 1).matmul(G) + num_total * weight_decay * torch.eye(G.shape[1], device=G.device)
        theta_inv = torch.linalg.inv(theta)

        w_complete = -G.matmul(theta_inv.matmul(f0_minus_y))

        theta_r = G_r.transpose(0, 1).matmul(G_r) + len(retain_loader.dataset) * weight_decay * torch.eye(G_r.shape[1], device=G_r.device)
        theta_r_inv = torch.linalg.inv(theta_r)
        w_retain = -G_r.matmul(theta_r_inv.matmul(f0_minus_y_r))

        return w_complete, w_retain

    def compute_delta_w(w_retain, w_complete):
        return (w_retain - w_complete).squeeze()

    def apply_scrubbing(model, delta_w, scale, beta):
        index = 0
        delta_w = delta_w.to(device)
        for param in model.parameters():
            num_params = param.numel()
            update = (beta * scale * delta_w[index:index + num_params]).view_as(param)
            param.data += update
            index += num_params

    # Copy and prepare models
    model_init = copy.deepcopy(model)

    for p in model.parameters():
        p.data0 = p.data.clone()

    # Compute NTK matrices and delta_w
    w_complete, w_retain = compute_ntk_matrices(model, retain_loader, forget_loader, num_classes, weight_decay)
    delta_w = compute_delta_w(w_retain, w_complete)
    delta_w = delta_w.to('cpu')
    w_complete = w_complete.to('cpu')
    w_retain = w_retain.to('cpu')

    print(f"Norm of w_complete: {torch.norm(w_complete)}")
    print(f"Norm of w_retain: {torch.norm(w_retain)}")
    print(f"Norm of delta_w: {torch.norm(delta_w)}")

    # Compute scale using the trapezium trick
    m_pred_error = vectorize_params(model) - vectorize_params(model_init) - w_retain.squeeze()

    inner = torch.dot(delta_w / torch.norm(delta_w), m_pred_error / torch.norm(m_pred_error))
    if inner < 0:
        angle = torch.arccos(inner) - torch.pi / 2
        predicted_norm = torch.norm(delta_w) + 2 * torch.sin(angle) * torch.norm(m_pred_error)
    else:
        angle = torch.arccos(inner)
        predicted_norm = torch.norm(delta_w) + 2 * torch.cos(angle) * torch.norm(m_pred_error)

    scale = predicted_norm / torch.norm(delta_w)

    print(f"Computed scale: {scale}")

    apply_scrubbing(model, delta_w, scale, beta)

    return model
