import warnings
warnings.filterwarnings('ignore')
import torch
import os
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
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
from auxil.auxils import *
from auxil.trivial_mi import *
from model.model import *


class InverterDataset(Dataset):
    def __init__(self, tensor1, tensor2, tensor3):
        assert tensor1.size(0) == tensor2.size(0) == tensor3.size(0)  # Ensure tensors have the same length
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.tensor3 = tensor3

    def __len__(self):
        return self.tensor1.size(0)  # Return the total number of samples

    def __getitem__(self, idx):
        sample_tensor1 = self.tensor1[idx]
        sample_tensor2 = self.tensor2[idx]
        sample_tensor3 = self.tensor3[idx]
        return sample_tensor1, sample_tensor2, sample_tensor3



class Inverter_Net(nn.Module):
    def __init__(self,beggining,end):
        super(Inverter_Net, self).__init__()
        self.beggining=beggining
        self.end=end

    def forward(self, x):
        x = self.beggining(x)
        x = self.end(x)
        return x
    


def condensive_inversion(directory, net, train_loader, img_shape, num_classes, random_classes, hidden_size, lr_inverter, inverter_epochs, batch_size, mi_epochs, case, device):

    images_dict = {class_name: [] for class_name in random_classes}

    net.to('cpu')

    for param in net.parameters():
        param.requires_grad = False
    
    soft_labels=[]
    hard_labels=[]
    imgs=[]
    for batch in train_loader:
        img,lab=batch
        out=F.softmax(net(img),dim=1)
        soft_labels.append(out)
        hard_labels.append(lab)
        imgs.append(img)

    soft_labels=torch.cat(soft_labels,dim=0)
    hard_labels=torch.cat(hard_labels,dim=0)
    imgs=torch.cat(imgs,dim=0)

    inversion_dataset=InverterDataset(soft_labels,hard_labels,imgs)
    inversion_loader=torch.utils.data.DataLoader(inversion_dataset, batch_size=batch_size, shuffle=True)

    inverted_net=InvertedMLP(input_size=num_classes, hidden_size=hidden_size, output_size=img_shape)

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
  
    for each_class in random_classes:

        one_hot = torch.zeros((1, num_classes))
        one_hot[0, each_class] = 1

        estim_img=inverter(one_hot.to(device)).detach().cpu().squeeze(0).numpy()
        images_dict[each_class].append(estim_img)


    file_path = os.path.join(directory,'SignleImgCondensiveMIAttack_'+str(case)+'.png')
    plot_images_matrix(images_dict, random_classes, figsize=(15, 15),filename=file_path)


