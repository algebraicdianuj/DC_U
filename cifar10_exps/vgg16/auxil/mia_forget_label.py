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
from art.estimators.classification.pytorch import PyTorchClassifier
from art.attacks.inference.membership_inference import LabelOnlyDecisionBoundary
from sklearn.metrics import accuracy_score

def label_only_mia(net,forget_loader, test_loader):

    x_test = []
    y_test = []


    for batch in test_loader:
        img,label = batch
        x_test.append(img)
        y_test.append(label)

    x_test = torch.cat(x_test, dim=0).numpy()
    y_test = torch.cat(y_test, dim=0).numpy().astype(np.uint8)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    art_model = PyTorchClassifier(model=net, loss=criterion, optimizer=optimizer, input_shape=(1,3,32,32), nb_classes=10)

    mia_label_only = LabelOnlyDecisionBoundary(art_model)

    forget_x=[]
    forget_y=[]

    for batch in forget_loader:
        img,label = batch
        forget_x.append(img)
        forget_y.append(label)

    forget_x = torch.cat(forget_x, dim=0).numpy()
    forget_y = torch.cat(forget_y, dim=0).numpy().astype(np.uint8)


    standard_len=len(forget_x)
    # standard_len=100
    sampling_index_test=np.random.choice(len(x_test), standard_len, replace=False)
    sampled_x_test=x_test[sampling_index_test]
    sampled_y_test=y_test[sampling_index_test]


    sampled_forget_index=np.random.choice(len(forget_x), standard_len, replace=False)
    sampled_forget_x=forget_x[sampled_forget_index]
    sampled_forget_y=forget_y[sampled_forget_index]


    x = np.concatenate([sampled_forget_x, sampled_x_test])
    y = np.concatenate([sampled_forget_y, sampled_y_test])
    training_sample = np.array([1] * len(sampled_forget_x) + [0] * len(sampled_x_test))

    mia_label_only.calibrate_distance_threshold(sampled_forget_x, sampled_forget_y,
                                                sampled_x_test, sampled_y_test,
                                                max_iter=5, max_eval=5,init_eval=1,init_size=1
                                                )


    x_eval, y_eval = sampled_forget_x, sampled_forget_y
    eval_label = np.array([1] * len(y_eval))

    pred_label = mia_label_only.infer(x_eval, y_eval,max_iter=5, max_eval=2,init_eval=1,init_size=1)

    return accuracy_score(eval_label, pred_label)*100

