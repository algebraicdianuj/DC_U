
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
import sys
from auxil.auxils import *
from auxil.auxils import *
from auxil.trivial_mi import *
from model.model import *
from auxil.innovative_mi_singleimg import *
from modular_unlearn.ds_condensation_imrpvDM import *
from auxil.innovative_mi_singleimg import *



def MI_condensation(new_lab_train, original_labels_dict, train_images, train_labels, net, n_classes,n_subclasses, img_shape,batch_inversion, lr_inverter, inverter_epochs, device):
    # one hot encode new labels
    soft_labels = torch.nn.functional.one_hot(new_lab_train.cpu(), num_classes=n_classes*n_subclasses).float()

    inversion_dataset=InverterDataset(soft_labels,train_labels,train_images)

    bucket_labbies=torch.unique(new_lab_train).tolist()

    net.to('cpu')

    inversion_loader=torch.utils.data.DataLoader(inversion_dataset, batch_size=batch_inversion, shuffle=True)

    inverted_net=InverterResnet18(output_size=img_shape[0]*img_shape[1]*img_shape[2]).to(device)

    combined_model=Inverter_Net(inverted_net,net).to(device)

    for param in combined_model.parameters():
        param.requires_grad = False

    for param in combined_model.beggining.parameters():
        param.requires_grad = True

    optim_combo=torch.optim.Adam(combined_model.parameters(), lr=lr_inverter)
    for _ in range(inverter_epochs):
        run_loss=0.0
        for batch_idx, (data, target, img) in enumerate(inversion_loader):
            data, target, img = data.to(device), target.to(device), img.to(device)
            inter_img=combined_model.beggining(data)
            output = combined_model.end(inter_img) 
            n_loss = nn.CrossEntropyLoss()(output, target)
            n_loss+= nn.MSELoss()(inter_img,img)
            optim_combo.zero_grad()
            n_loss.backward()
            optim_combo.step()
            run_loss+=n_loss.item()

        if _%(int(inverter_epochs/5))==0 or _==inverter_epochs-1 or _==0:
            print('Inversion Epoch: {} \tLoss: {:.6f}'.format(_, run_loss / len(inversion_loader)))


    inverter=combined_model.beggining

    inverted_IMG=[]
    inverted_LABEL=[]
    indices_train_wrt_bucket=[]

    for lab in bucket_labbies:
        one_hot = torch.zeros((1, n_classes*n_subclasses))
        one_hot[0, lab] = 1
        estim_img=inverter(one_hot.to(device)).detach().cpu()
        inverted_IMG.append(estim_img)
        inverted_LABEL.append(original_labels_dict[lab])
        indices_lab = torch.where(new_lab_train.to(device)==lab)[0]
        indices_train_wrt_bucket.append(indices_lab.cpu())
        
    inverted_IMG=torch.cat(inverted_IMG, dim=0).cpu()
    inverted_LABEL=torch.tensor(inverted_LABEL).cpu()


    return inverted_IMG, inverted_LABEL, indices_train_wrt_bucket, bucket_labbies
