import warnings
warnings.filterwarnings('ignore')
import os
import torch
import torch.nn as nn
import numpy as np
import time
import copy
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
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pickle
from torchvision.models import resnet18

from modular_unlearn.offline_training import *
from modular_unlearn.ds_condensation_imrpvDM import *
from modular_unlearn.modular_forgetting import *
from modular_unlearn.unlearning_metric import *
from modular_unlearn.overfitting_metric import *
from auxil.auxils import *
from model.model import *
from auxil.retrain import *
from auxil.distillation import *
from auxil.sparisification import *
from auxil.bad_distillation import *
from modular_unlearn.ds_condensation_MI import *


def main():

    directory_name= 'reservoir'
    current_path = os.getcwd()  
    new_directory_path = os.path.join(current_path, directory_name)  
    
    if not os.path.exists(new_directory_path): 
        os.makedirs(new_directory_path) 
        print(f"Directory '{directory_name}' created in the current working directory.")
    else:
        print(f"Directory '{directory_name}' already exists in the current working directory.")

    #----------------------Hyperparameters---------------------------------
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    batch_size_bucket=128
    batch_syn=32
    channel = 3
    im_size = (32, 32)

    num_classes = 10
    batch_real = 5000
    split_ratio = 0.1   # forget-retain split ratio
    n_subclasses= 45   # K-means "K"
    offline_condensation_iterations = 10    #10
    final_model_epochs = 20
    databank_model_epochs = 20
    lr_final=1e-4
    lr_databank=1e-5
    condensation_epochs=100   #100

    batch_inversion=128
    lr_inverter=1e-3
    inverter_epochs=200

    #------------------------------------------------------------------------


    #----------------------------Loading stuff------------------------------------------------------------------------
    vgg16=modify_vgg16(channel, im_size[0], num_classes)
    net=Vgg16(vgg16=vgg16).to(device)
    file_path = os.path.join(new_directory_path,'pretrained_net.pth')
    net.load_state_dict(torch.load(file_path))

    file_path = os.path.join(new_directory_path,'Klabels_labels_dict.pkl')
    with open(file_path, 'rb') as file:
        original_labels_dict = pickle.load(file)


    file_path = os.path.join(new_directory_path,'forget_set.pth')
    forget_set_real = torch.load(file_path)
    file_path = os.path.join(new_directory_path,'retain_set.pth')
    retain_set_real = torch.load(file_path)
    file_path = os.path.join(new_directory_path,'test_set.pth')
    dst_test = torch.load(file_path)
    file_path = os.path.join(new_directory_path,'indices.pth')
    indices = torch.load(file_path)
    file_path = os.path.join(new_directory_path,'forget_indices.pth')
    forget_indices = torch.load(file_path)
    file_path = os.path.join(new_directory_path,'retain_indices.pth')
    retain_indices = torch.load(file_path)
    file_path = os.path.join(new_directory_path,'clustered_label_train.pth')
    new_lab_train = torch.load(file_path)
    file_path = os.path.join(new_directory_path,'image_train.pth')
    train_images = torch.load(file_path)
    file_path = os.path.join(new_directory_path,'label_train.pth')
    train_labels = torch.load(file_path)
    file_path = os.path.join(new_directory_path,'syn_set.pth')
    img_syn_dataset = torch.load(file_path)

    img_real_data_dataset=TensorDatasett(train_images, train_labels)
    img_real_data_loader=torch.utils.data.DataLoader(img_real_data_dataset, batch_size=batch_size, shuffle=True)
    forget_loader=torch.utils.data.DataLoader(forget_set_real, batch_size=batch_size, shuffle=True)
    retain_loader=torch.utils.data.DataLoader(retain_set_real, batch_size=batch_size, shuffle=True)
    img_syn_loader=torch.utils.data.DataLoader(img_syn_dataset, batch_size=batch_syn, shuffle=True)
    test_loader=torch.utils.data.DataLoader(dst_test, batch_size=batch_size, shuffle=True)



    #-----------------------------Offline training-----------------------------------------------------------------------
    print("\n--Starting offline training of modularized network--\n")
    offline_trainer(new_directory_path=new_directory_path, net=net, img_real_data_loader=img_real_data_loader, img_syn_loader=img_syn_loader, retain_loader=retain_loader, forget_loader=forget_loader, device=device, offline_condensation_iterations=offline_condensation_iterations, final_model_epochs=final_model_epochs, databank_model_epochs=databank_model_epochs, lr_final=lr_final, lr_databank=lr_databank)
    #-------------------------------------------------------------------------------------------------------------------


    #-----------------------------Dataset Condensation and Offline data processing--------------------------------------------------------------
    inverted_IMG, inverted_LABEL, indices_train_wrt_bucket, bucket_labbies= MI_condensation(new_lab_train=new_lab_train, original_labels_dict=original_labels_dict, train_images=train_images, train_labels=train_labels, net=net, n_classes=num_classes,n_subclasses=n_subclasses, img_shape=(channel,)+im_size,batch_inversion=batch_inversion, lr_inverter=lr_inverter, inverter_epochs=inverter_epochs, device=device)
    # inverted_IMG, inverted_LABEL, indices_train_wrt_bucket, bucket_labbies= improv_DM_condensation(new_lab_train=new_lab_train, train_images=train_images, train_labels=train_labels, net=net, condensation_epochs=condensation_epochs, n_classes=num_classes,n_subclasses=n_subclasses, device=device)
    reduced_imgs,reduced_labs= offline_data_processing(inverted_IMG, inverted_LABEL,  indices_train_wrt_bucket, forget_indices, retain_indices, bucket_labbies, img_real_data_dataset,train_images,train_labels,forget_loader, retain_loader,device)
    #-------------------------------------------------------------------------------------------------------------------------------------------

    file_path = os.path.join(new_directory_path,'reduced_retain_images.pth')
    torch.save(reduced_imgs, file_path)
    file_path = os.path.join(new_directory_path,'reduced_retain_labels.pth')
    torch.save(reduced_labs, file_path)
    file_path = os.path.join(new_directory_path,'inverted_IMG.pth')
    torch.save(inverted_IMG, file_path)
    file_path = os.path.join(new_directory_path,'inverted_LABEL.pth')
    torch.save(inverted_LABEL, file_path)
    file_path = os.path.join(new_directory_path,'indices_train_wrt_bucket.pth')
    torch.save(indices_train_wrt_bucket, file_path)
    file_path = os.path.join(new_directory_path,'bucket_labbies.pth')
    torch.save(bucket_labbies, file_path)



if __name__ == '__main__':
    main()






