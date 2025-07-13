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
from auxil.innovative_mi_singleimg import *



def condensive_inversion(directory, net, ref_net, training_images, training_labels, img_shape, num_classes, random_classes, samps_per_class, hidden_size, lr_inverter, inverter_epochs, batch_size, case, device):

    new_lab_train, original_labels_dict = create_sub_classes(training_images, training_labels, model=ref_net, num_classes=num_classes, sub_divisions=samps_per_class)
     # one hot encode new labels
    soft_labels = torch.nn.functional.one_hot(new_lab_train.cpu(), num_classes=num_classes*samps_per_class).float()

    inversion_dataset=InverterDataset(soft_labels,training_labels,training_images)

    images_dict = {class_name: [] for class_name in random_classes}

    net.to('cpu')

    inversion_loader=torch.utils.data.DataLoader(inversion_dataset, batch_size=batch_size, shuffle=True)

    inverted_net=InvertedMLP(input_size=num_classes*samps_per_class, hidden_size=hidden_size, output_size=img_shape)

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
        # find all keys with valye each_class
        keys_each_class = [k for k, v in original_labels_dict.items() if v == each_class]

        for keys in keys_each_class:
            one_hot = torch.zeros((1, num_classes*samps_per_class))
            one_hot[0, keys] = 1

            estim_img=inverter(one_hot.to(device)).detach().cpu().squeeze(0).numpy()
            images_dict[each_class].append(estim_img)


    file_path = os.path.join(directory,'MutiImgCondensiveMIAttack_'+str(case)+'.png')
    plot_images_matrix(images_dict, random_classes, figsize=(15, 15),filename=file_path)


