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
from auxil.auxils import *
from model.model import *
from auxil.retrain import *
from auxil.distillation import *
from auxil.sparisification import *
from auxil.bad_distillation import *
from auxil.mia_forget_logit import *
from modular_unlearn.overfitting_metric import *


def main():
    directory_name= 'reservoir'
    current_path = os.getcwd()  
    new_directory_path = os.path.join(current_path, directory_name)  
    
    if not os.path.exists(new_directory_path): 
        os.makedirs(new_directory_path) 
        print(f"Directory '{directory_name}' created in the current working directory.")
    else:
        print(f"Directory '{directory_name}' already exists in the current working directory.")


    dat_dir='result'
    result_directory_path = os.path.join(current_path, dat_dir)

    if not os.path.exists(result_directory_path):
        os.makedirs(result_directory_path)
        print(f"Directory '{dat_dir}' created in the current working directory.")
    else:
        print(f"Directory '{dat_dir}' already exists in the current working directory.")

    #----------------------Hyperparameters---------------------------------
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    batch_syn=32
    channel = 3
    im_size = (32, 32)
    hidden_size=128
    num_classes = 10
    lr_proposed=1e-3



    lr_overture=1e-3
    lr_intermediate=2e-3
    overture_epochs=1
    beggining_epochs=1
    final_epochs = 80
    intermediate_epochs= 1
    final_thr= 5  # intended for blocking the final training in overture, 
                # from the end of overture epochs--> improves retain acc while preserving forget accuracy

    threshold = 0.5  # Choose an appropriate threshold for binarizing the Fisher Information
    lambd=0.1   #noise addition magnitude


    retrain_lr=1e-3
    retrain_epochs=1


    criterion = nn.CrossEntropyLoss()
    #------------------------------------------------------------------------


    #----------------------------Loading stuff------------------------------------------------------------------------
    net=vitl(im_size=im_size[0], num_classes=num_classes).to(device)
    ref_net=copy.deepcopy(net)
    for param in list(ref_net.parameters()):
        param.requires_grad = False
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
    file_path = os.path.join(new_directory_path,'reduced_retain_images.pth')
    reduced_retain_images=torch.load(file_path)
    file_path = os.path.join(new_directory_path,'reduced_retain_labels.pth')
    reduced_retain_labels=torch.load(file_path)
    file_path = os.path.join(new_directory_path,'inverted_IMG.pth')
    inverted_IMG=torch.load(file_path)
    file_path = os.path.join(new_directory_path,'inverted_LABEL.pth')
    inverted_LABEL=torch.load(file_path)
    file_path = os.path.join(new_directory_path,'indices_train_wrt_bucket.pth')
    indices_train_wrt_bucket=torch.load(file_path)
    file_path = os.path.join(new_directory_path,'bucket_labbies.pth')
    bucket_labbies=torch.load(file_path)


    img_real_data_dataset=TensorDatasett(train_images, train_labels)
    img_real_data_loader=torch.utils.data.DataLoader(img_real_data_dataset, batch_size=batch_size, shuffle=True)
    forget_loader=torch.utils.data.DataLoader(forget_set_real, batch_size=batch_size, shuffle=True)
    retain_loader=torch.utils.data.DataLoader(retain_set_real, batch_size=batch_size, shuffle=True)
    img_syn_loader=torch.utils.data.DataLoader(img_syn_dataset, batch_size=batch_syn, shuffle=True)
    test_loader=torch.utils.data.DataLoader(dst_test, batch_size=batch_size, shuffle=True)
    reduced_retain_dataset=TensorDatasett(reduced_retain_images,reduced_retain_labels)
    reduced_retain_loader=torch.utils.data.DataLoader(reduced_retain_dataset, batch_size=batch_size, shuffle=True)


    #---------------------------------------Pretrained Model--------------------------------------------------------------------------
    mia_score=measure_mia(net, forget_loader,test_loader, device)
    retain_acc=test(net, retain_loader, device)
    forget_acc=test(net, forget_loader, device)
    unlearning_metric=measure_unlearning(net, forget_loader, retain_loader, device)
    overfitting_metric=measure_overfitting(net, criterion, forget_loader, device)


    print('\nPretrained Unlearning Stats: ')
    print('======================================')
    print('mia_score: ', mia_score)
    print('retain_acc: ', retain_acc)
    print('forget_acc: ', forget_acc)
    print('unlearning_metric: ', unlearning_metric)
    print('overfitting_metric: ', overfitting_metric)
    print('======================================')

    stat_data = {
        'mia_score': [mia_score],
        'retain_acc': [retain_acc],
        'forget_acc': [forget_acc],
        'unlearning_metric': [unlearning_metric],
        'overfitting_metric': [overfitting_metric]
    }

    df = pd.DataFrame(stat_data)

    file_path = os.path.join(result_directory_path,'pretrained_unlearning_stats.csv')
    df.to_csv(file_path, index=False)



    #--------------------------Initializing my Unlearning Method--------------------------------------------------------------------------- 
    beggining=Beginning(ref_net).to(device)

    intermediate=Intermediate(ref_net).to(device)

    data_bank=Databank(beggining=beggining, intermediate=intermediate).to(device)
    file_path = os.path.join(new_directory_path,'databank.pth')
    data_bank.load_state_dict(torch.load(file_path))

    final=Final(ref_net).to(device)
    file_path = os.path.join(new_directory_path,'final.pth')
    final.load_state_dict(torch.load(file_path))

    combined_model=CombinedModel(databank=data_bank, final=final).to(device)

    optim_model=torch.optim.Adam(combined_model.parameters(), lr=lr_proposed)
    criterion = nn.CrossEntropyLoss()
#----------------------------------------------------------------------------------------------------------------------------------


#--------------------------my Unlearning Method------------------------------------------------------------------------------------

    starting_time = time.time()
    combined_model=modular_unlearning(combined_model, optim_model, lr_overture,lr_intermediate,criterion, device, beggining_epochs, intermediate_epochs, final_epochs, overture_epochs, final_thr, img_syn_loader, reduced_retain_loader)
    ending_time = time.time()
    unlearning_time=ending_time - starting_time
    mia_score=measure_mia(combined_model, forget_loader,test_loader, device)
    retain_acc=test(combined_model, retain_loader, device)
    forget_acc=test(combined_model, forget_loader, device)
    unlearning_metric=measure_unlearning(combined_model, forget_loader, retain_loader, device)
    overfitting_metric=measure_overfitting(combined_model, criterion, forget_loader, device)


    print('\nModular Unlearning Stats: ')
    print('======================================')
    print('mia_score: ', mia_score)
    print('retain_acc: ', retain_acc)
    print('forget_acc: ', forget_acc)
    print('unlearning_metric: ', unlearning_metric)
    print('overfitting_metric: ', overfitting_metric)
    print('======================================')

    stat_data = {
        'mia_score': [mia_score],
        'retain_acc': [retain_acc],
        'forget_acc': [forget_acc],
        'unlearning_metric': [unlearning_metric],
        'overfitting_metric': [overfitting_metric]
    }

    df = pd.DataFrame(stat_data)

    file_path = os.path.join(result_directory_path,'modular_unlearning_stats_epochs'+str(overture_epochs)+'.csv')
    df.to_csv(file_path, index=False)

   #----------------------Catastrophic Forgetting Method---------------------------------------------------------------------------------
    naive_net=vitl(im_size=im_size[0], num_classes=num_classes).to(device)
    file_path = os.path.join(new_directory_path,'pretrained_net.pth')
    naive_net.load_state_dict(torch.load(file_path))
    starting_time = time.time()
    naive_net= retraining(naive_net, criterion, device, retrain_lr, retrain_epochs, reduced_retain_loader)
    ending_time = time.time()
    unlearning_time=ending_time - starting_time

    mia_score=measure_mia(naive_net, forget_loader,test_loader, device)
    retain_acc=test(naive_net, retain_loader, device)
    forget_acc=test(naive_net, forget_loader, device)
    unlearning_metric=measure_unlearning(naive_net, forget_loader, retain_loader, device)
    overfitting_metric=measure_overfitting(naive_net, criterion, forget_loader, device)


    print('\nCatastrophic-Forgetting Stats: ')
    print('======================================')
    print('mia_score: ', mia_score)
    print('retain_acc: ', retain_acc)
    print('forget_acc: ', forget_acc)
    print('unlearning_metric: ', unlearning_metric)
    print('overfitting_metric: ', overfitting_metric)
    print('======================================')

    stat_data = {
        'mia_score': [mia_score],
        'retain_acc': [retain_acc],
        'forget_acc': [forget_acc],
        'unlearning_metric': [unlearning_metric],
        'overfitting_metric': [overfitting_metric]
    }

    df = pd.DataFrame(stat_data)
    file_path = os.path.join(result_directory_path,'catastrophic_unlearning_stats_epochs'+str(retrain_epochs)+'.csv')
    df.to_csv(file_path, index=False)



if __name__ == '__main__':
    main()