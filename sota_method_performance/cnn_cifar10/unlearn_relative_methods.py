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
from auxil.ntk_scrubbing import *
from auxil.fisher_forget2 import *


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
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    batch_syn=32
    channel = 3
    im_size = (32, 32)
    hidden_size=128
    num_classes = 10
    lr_proposed=1e-3


    retrain_lr=1e-3
    retrain_epochs=50

    distill_lr=1e-3
    distill_epochs=50
    hard_weigth=1
    soft_weight=1e-1
    kdT=4.0

    no_l1_epochs=15
    weight_l1=1e-4
    pruning_ratio = 0.05
    partial_retain_ratio=0.3
    #------------------------------------------------------------------------


    #----------------------------Loading stuff------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    net= CNN(channel, im_size[0], num_classes).to(device)
    file_path = os.path.join(new_directory_path,'pretrained_net.pth')
    net.load_state_dict(torch.load(file_path))



    file_path = os.path.join(new_directory_path,'forget_set.pth')
    forget_set_real = torch.load(file_path)
    file_path = os.path.join(new_directory_path,'retain_set.pth')
    retain_set_real = torch.load(file_path)

    file_path = os.path.join(new_directory_path,'test_set.pth')
    dst_test = torch.load(file_path)

    forget_loader=torch.utils.data.DataLoader(forget_set_real, batch_size=batch_size, shuffle=True)
    retain_loader=torch.utils.data.DataLoader(retain_set_real, batch_size=batch_size, shuffle=True)
    test_loader=torch.utils.data.DataLoader(dst_test, batch_size=batch_size, shuffle=True)


    # sample the retain_set_real by partial_retain_ration
    partial_retain_set_real = torch.utils.data.Subset(retain_set_real, random.sample(range(len(retain_set_real)), int(len(retain_set_real)*partial_retain_ratio)))



    #--------------------------Initializing--------------------------------------------------------------------------- 
    ref_net=copy.deepcopy(net)
    for param in list(ref_net.parameters()):
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    #----------------------------------------------------------------------------------------------------------------------------------




    #--------------------------Retraining Method------------------------------------------------------------------------------------
    naive_net= CNN(channel, im_size[0], num_classes).to(device)
    starting_time = time.time()
    naive_net= retraining(naive_net, criterion, device, retrain_lr,retrain_epochs, retain_loader)
    ending_time = time.time()
    unlearning_time=ending_time - starting_time

    mia_score=measure_mia(naive_net, forget_loader,test_loader, device)
    retain_acc=test(naive_net, retain_loader, device)
    forget_acc=test(naive_net, forget_loader, device)

    print('\nRetraining-Unlearning Stats: ')
    print('======================================')
    print('mia_score: ', mia_score)
    print('retain_acc: ', retain_acc)
    print('forget_acc: ', forget_acc)
    print('unlearning_time: ', unlearning_time)
    print('======================================')

    stat_data = {
        'mia_score': [mia_score],
        'retain_acc': [retain_acc],
        'forget_acc': [forget_acc],
        'unlearning_time': [unlearning_time]
    }

    df = pd.DataFrame(stat_data)

    file_path = os.path.join(result_directory_path,'retraining_unlearning_stats.csv')
    df.to_csv(file_path, index=False)

    del naive_net


    #----------------------Catastrophic Forgetting Method---------------------------------------------------------------------------------
    naive_net= CNN(channel, im_size[0], num_classes).to(device)
    file_path = os.path.join(new_directory_path,'pretrained_net.pth')
    naive_net.load_state_dict(torch.load(file_path))
    starting_time = time.time()
    naive_net= retraining(naive_net, criterion, device, retrain_lr, retrain_epochs, retain_loader)
    ending_time = time.time()
    unlearning_time=ending_time - starting_time

    mia_score=measure_mia(naive_net, forget_loader,test_loader, device)
    retain_acc=test(naive_net, retain_loader, device)
    forget_acc=test(naive_net, forget_loader, device)


    print('\nCatastrophic-Forgetting Stats: ')
    print('======================================')
    print('mia_score: ', mia_score)
    print('retain_acc: ', retain_acc)
    print('forget_acc: ', forget_acc)
    print('unlearning_time: ', unlearning_time)
    print('======================================')

    stat_data = {
        'mia_score': [mia_score],
        'retain_acc': [retain_acc],
        'forget_acc': [forget_acc],
        'unlearning_time': [unlearning_time]
    }

    df = pd.DataFrame(stat_data)

    file_path = os.path.join(result_directory_path,'catastrophic_unlearning_stats.csv')
    df.to_csv(file_path, index=False)

    del naive_net



    #----------------------Fischer-Forgetting Method---------------------------------------------------------------------------------
    naive_net= CNN(channel, im_size[0], num_classes).to(device)
    file_path = os.path.join(new_directory_path,'pretrained_net.pth')
    naive_net.load_state_dict(torch.load(file_path))
    starting_time = time.time()
    forgot_net=fischer_forgetting(naive_net, retain_loader, num_classes, device, class_to_forget=None, num_to_forget=None,alpha=1e-7)
    ending_time = time.time()
    unlearning_time=ending_time - starting_time

    mia_score=measure_mia(forgot_net, forget_loader, test_loader, device)
    retain_acc=test(forgot_net, retain_loader, device)
    forget_acc=test(forgot_net, forget_loader, device)

    print('\nFisher-Forgetting Stats: ')
    print('======================================')
    print('mia_score: ', mia_score)
    print('retain_acc: ', retain_acc)
    print('forget_acc: ', forget_acc)
    print('unlearning_time: ', unlearning_time)
    print('======================================')

    stat_data = {
        'mia_score': [mia_score],
        'retain_acc': [retain_acc],
        'forget_acc': [forget_acc],
        'unlearning_time': [unlearning_time]
    }

    df = pd.DataFrame(stat_data)

    file_path = os.path.join(result_directory_path,'fisher_forgetting_stats.csv')
    df.to_csv(file_path, index=False)

    del forgot_net
    del naive_net


    #----------------------NTK-Scrubbing Method---------------------------------------------------------------------------------
    naive_net= CNN(channel, im_size[0], num_classes).to(device)
    file_path = os.path.join(new_directory_path,'pretrained_net.pth')
    naive_net.load_state_dict(torch.load(file_path))
    starting_time = time.time()
    forgot_net=ntk_scrubbing(naive_net, retain_loader, forget_loader,  device=device, num_classes=num_classes, weight_decay=1e-1, beta=1)
    ending_time = time.time()
    unlearning_time=ending_time - starting_time

    mia_score=measure_mia(forgot_net, forget_loader, test_loader, device)
    retain_acc=test(forgot_net, retain_loader, device)
    forget_acc=test(forgot_net, forget_loader, device)

    print('\nNTK-Scrubbing Stats: ')
    print('======================================')
    print('mia_score: ', mia_score)
    print('retain_acc: ', retain_acc)
    print('forget_acc: ', forget_acc)
    print('unlearning_time: ', unlearning_time)
    print('======================================')

    stat_data = {
        'mia_score': [mia_score],
        'retain_acc': [retain_acc],
        'forget_acc': [forget_acc],
        'unlearning_time': [unlearning_time]
    }

    df = pd.DataFrame(stat_data)

    file_path = os.path.join(result_directory_path,'ntk_scrubbing_stats.csv')
    df.to_csv(file_path, index=False)

    del forgot_net
    del naive_net




    #-----------------------Distillation based Method---------------------------------------------------------------------------------
    teacher_net= CNN(channel, im_size[0], num_classes).to(device)
    file_path = os.path.join(new_directory_path,'pretrained_net.pth')
    teacher_net.load_state_dict(torch.load(file_path))
    student_net= CNN(channel, im_size[0], num_classes).to(device)
    starting_time = time.time()
    distilled_net = distillation_unlearning(retain_loader, distill_lr ,student_net, teacher_net, distill_epochs, device, alpha=hard_weigth, gamma=soft_weight, kd_T=kdT)
    ending_time = time.time()
    unlearning_time=ending_time - starting_time

    mia_score=measure_mia(distilled_net, forget_loader, test_loader, device)
    retain_acc=test(distilled_net, retain_loader, device)
    forget_acc=test(distilled_net, forget_loader, device)



    print('\nDistillation-Unlearning Stats: ')
    print('======================================')
    print('mia_score: ', mia_score)
    print('retain_acc: ', retain_acc)
    print('forget_acc: ', forget_acc)
    print('unlearning_time: ', unlearning_time)
    print('======================================')

    stat_data = {
        'mia_score': [mia_score],
        'retain_acc': [retain_acc],
        'forget_acc': [forget_acc],
        'unlearning_time': [unlearning_time]
    }

    df = pd.DataFrame(stat_data)

    file_path = os.path.join(result_directory_path,'distillation_unlearning_stats.csv')
    df.to_csv(file_path, index=False)

    del distilled_net
    del teacher_net
    del student_net




    #-----------------------Bad Teacher based Method---------------------------------------------------------------------------------
    teacher_net= CNN(channel, im_size[0], num_classes).to(device)
    file_path = os.path.join(new_directory_path,'pretrained_net.pth')
    teacher_net.load_state_dict(torch.load(file_path))
    student_net= CNN(channel, im_size[0], num_classes).to(device)
    file_path = os.path.join(new_directory_path,'pretrained_net.pth')
    student_net.load_state_dict(torch.load(file_path))
    bad_teacher_net= CNN(channel, im_size[0], num_classes).to(device)
    starting_time = time.time()
    distilled_net=blindspot_unlearner(model=student_net, unlearning_teacher=bad_teacher_net, full_trained_teacher=teacher_net, retain_data=partial_retain_set_real,forget_data=forget_set_real, epochs = distill_epochs,
                        lr = distill_lr, batch_size = batch_size, device = device, KL_temperature = kdT)
    
    ending_time = time.time()
    unlearning_time=ending_time - starting_time

    mia_score=measure_mia(distilled_net, forget_loader, test_loader, device)
    retain_acc=test(distilled_net, retain_loader, device)
    forget_acc=test(distilled_net, forget_loader, device)

    print('\nBad Teacher Distillation-Unlearning Stats: ')
    print('======================================')
    print('mia_score: ', mia_score)
    print('retain_acc: ', retain_acc)
    print('forget_acc: ', forget_acc)
    print('unlearning_time: ', unlearning_time)
    print('======================================')

    stat_data = {
        'mia_score': [mia_score],
        'retain_acc': [retain_acc],
        'forget_acc': [forget_acc],
        'unlearning_time': [unlearning_time]
    }

    df = pd.DataFrame(stat_data)

    file_path = os.path.join(result_directory_path,'bad_teacher_distillation_unlearning_stats.csv')
    df.to_csv(file_path , index=False)

    del distilled_net
    del teacher_net
    del student_net
    del bad_teacher_net




    #-----------------------Sparsification based Method---------------------------------------------------------------------------------
    naive_net= CNN(channel, im_size[0], num_classes).to(device)
    file_path = os.path.join(new_directory_path,'pretrained_net.pth')
    naive_net.load_state_dict(torch.load(file_path))
    starting_time = time.time()
    sparsified_net = sparse_unlearning(retain_loader, naive_net, retrain_lr, retrain_epochs, no_l1_epochs=no_l1_epochs, alpha=weight_l1, device=device)
    ending_time = time.time()
    unlearning_time=ending_time - starting_time

    mia_score=measure_mia(sparsified_net, forget_loader, test_loader, device)
    retain_acc=test(sparsified_net, retain_loader, device)
    forget_acc=test(sparsified_net, forget_loader, device)

    print('\nSparsification-Unlearning Stats: ')
    print('======================================')
    print('mia_score: ', mia_score)
    print('retain_acc: ', retain_acc)
    print('forget_acc: ', forget_acc)
    print('unlearning_time: ', unlearning_time)
    print('======================================')

    stat_data = {
        'mia_score': [mia_score],
        'retain_acc': [retain_acc],
        'forget_acc': [forget_acc],
        'unlearning_time': [unlearning_time]
    }

    df = pd.DataFrame(stat_data)

    file_path = os.path.join(result_directory_path,'sparsification_unlearning_stats.csv')
    df.to_csv(file_path, index=False)

    del sparsified_net
    del naive_net



    #-----------------------Pruning based Method---------------------------------------------------------------------------------
    naive_net= CNN(channel, im_size[0], num_classes).to(device)
    file_path = os.path.join(new_directory_path,'pretrained_net.pth')
    naive_net.load_state_dict(torch.load(file_path))
    starting_time = time.time()
    pruned_net=prune_and_retrain(naive_net, criterion, device, retrain_lr, retrain_epochs, retain_loader, prune_ratio = 0.95)
    ending_time = time.time()
    unlearning_time=ending_time - starting_time

    mia_score=measure_mia(pruned_net, forget_loader, test_loader, device)
    retain_acc=test(pruned_net, retain_loader, device)
    forget_acc=test(pruned_net, forget_loader, device)

    print('\nPrunning-Unlearning Stats: ')
    print('======================================')
    print('mia_score: ', mia_score)
    print('retain_acc: ', retain_acc)
    print('forget_acc: ', forget_acc)
    print('unlearning_time: ', unlearning_time)
    print('======================================')

    stat_data = {
        'mia_score': [mia_score],
        'retain_acc': [retain_acc],
        'forget_acc': [forget_acc],
        'unlearning_time': [unlearning_time]
    }

    df = pd.DataFrame(stat_data)

    file_path = os.path.join(result_directory_path,'prunning_unlearning_stats.csv')
    df.to_csv(file_path, index=False)

    del pruned_net
    del naive_net


if __name__ == '__main__':
    main()