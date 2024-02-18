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
import torch.nn.utils.prune as prune


def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )



def testing_losses(model, distill_loader, device):
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction='none')

    losses = []

    with torch.no_grad():
        for inputs, labels in distill_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.detach().cpu().numpy())

    losses = np.concatenate(losses)
    return losses



def measure_mia(model, forget_loader, test_loader, device):
    forget_losses=testing_losses(model, forget_loader, device)
    test_losses=testing_losses(model, test_loader, device)

    stack_size=min([len(forget_losses), len(test_losses)])
    forget_losses = forget_losses[: stack_size]
    test_losses = test_losses[: stack_size]


    samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(test_losses) + [1] * len(forget_losses)

    mia_cands=[]
    for i in range(20):
        mia_scores = simple_mia(samples_mia, labels_mia)
        mia_cands.append(mia_scores.mean())

    mia_score=np.mean(mia_cands)

    return mia_score*100