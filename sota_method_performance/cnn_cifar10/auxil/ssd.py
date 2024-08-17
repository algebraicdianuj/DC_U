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



def zerolike_params_dict(model,device):
    return dict(
        [
            (k,torch.zeros_like(p,device=device)) for k,p in model.named_parameters()
        ]
    )


def calculate_importance(model,dataloader,device):
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    importances=zerolike_params_dict(model,device)
    for batch in dataloader:
        x,y=batch
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad()
        output=model(x)
        loss=criterion(output,y)
        loss.backward()

        for (k1,p),(k2,imp) in zip (model.named_parameters(),importances.items()):
            if p.grad is not None:
                imp.data+=p.grad.data.clone().pow(2)

    for _,imp in importances.items():
        imp.data/=float(len(dataloader))


    return importances


def modify_weights(model,original_importance,forget_importance,selection_weighting,dampening_constant,exponent=1, lower_bound=1):
    with torch.no_grad():
        for (n,p),(oimp_n,oimp),(fimp_n,fimp) in zip(model.named_parameters(), original_importance.items(), forget_importance.items()):
            oimp_norm=oimp.mul(selection_weighting)
            locations=torch.where(fimp>oimp_norm)
            weight=((oimp.mul(dampening_constant)).div(fimp)).pow(exponent)
            update=weight[locations]
            min_locs=torch.where(update>lower_bound)
            update[min_locs]=lower_bound
            p[locations]=p[locations].mul(update)



def ssd_unlearn(model, full_loader, forget_loader, device, selection_weighting,dampening_constant, exponent=1, lower_bound=1):
    sample_importances=calculate_importance(model,forget_loader,device)
    original_importances=calculate_importance(model,full_loader,device)
    modify_weights(model,original_importances,sample_importances, selection_weighting,dampening_constant, exponent, lower_bound)
    return model 
