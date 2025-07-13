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
    lr_proposed=0.005

    overture_epochs=10
    beggining_epochs=2
    final_epochs = 50
    intermediate_epochs= 5
    final_thr=1   # intended for blocking the final training in overture, 
                # from the end of overture epochs--> improves retain acc while preserving forget accuracy

    threshold = 0.5  # Choose an appropriate threshold for binarizing the Fisher Information
    lambd=0.1   #noise addition magnitude


    retrain_lr=1e-3
    retrain_epochs=30

    num_cycles=5
    #------------------------------------------------------------------------


    #----------------------------Loading stuff------------------------------------------------------------------------
    forgetter=[]
    retainer=[]
    reduced_retainer=[]

    for cycle in range(num_cycles):
        file_path = os.path.join(new_directory_path,'forget_set'+str(cycle)+'.pth')
        forget_set_real = torch.load(file_path)
        file_path = os.path.join(new_directory_path,'retain_set'+str(cycle)+'.pth')
        retain_set_real = torch.load(file_path)
        file_path = os.path.join(new_directory_path,'reduced_retain_images'+str(cycle)+'.pth')
        reduced_retain_images=torch.load(file_path)
        file_path = os.path.join(new_directory_path,'reduced_retain_labels'+str(cycle)+'.pth')
        reduced_retain_labels=torch.load(file_path)
        reduced_retain_dataset=TensorDatasett(reduced_retain_images,reduced_retain_labels)


        forget_loader=torch.utils.data.DataLoader(forget_set_real, batch_size=batch_size, shuffle=True)
        retain_loader=torch.utils.data.DataLoader(retain_set_real, batch_size=batch_size, shuffle=True)
        reduced_retain_loader=torch.utils.data.DataLoader(reduced_retain_dataset, batch_size=batch_size, shuffle=True)

        forgetter.append(forget_loader)
        retainer.append(retain_loader)
        reduced_retainer.append(reduced_retain_loader)

    file_path = os.path.join(new_directory_path,'test_set.pth')
    dst_test = torch.load(file_path)
    file_path = os.path.join(new_directory_path,'image_train.pth')
    train_images = torch.load(file_path)
    file_path = os.path.join(new_directory_path,'label_train.pth')
    train_labels = torch.load(file_path)
    file_path = os.path.join(new_directory_path,'syn_set.pth')
    img_syn_dataset = torch.load(file_path)




    img_real_data_dataset=TensorDatasett(train_images, train_labels)
    img_real_data_loader=torch.utils.data.DataLoader(img_real_data_dataset, batch_size=batch_size, shuffle=True)
    img_syn_loader=torch.utils.data.DataLoader(img_syn_dataset, batch_size=batch_syn, shuffle=True)
    test_loader=torch.utils.data.DataLoader(dst_test, batch_size=batch_size, shuffle=True)


    unlearning_time=[]
    mia_score=[]
    retain_acc=[]
    forget_acc=[]
    test_acc=[]


    net= CNN(channel, im_size[0], num_classes).to(device)
    for cycle in range(num_cycles):
        #--------------------------Initializing my Unlearning Method--------------------------------------------------------------------------- 
        beggining=Beginning(cnn=net).to(device)
        intermediate=Intermediate(cnn=net).to(device)
        final=Final(cnn=net).to(device)

        data_bank=Databank(beggining=beggining, intermediate=intermediate).to(device)
        file_path = os.path.join(new_directory_path,'databank.pth')
        data_bank.load_state_dict(torch.load(file_path))

        file_path = os.path.join(new_directory_path,'final.pth')
        final.load_state_dict(torch.load(file_path))
        combined_model=CombinedModel(databank=data_bank, final=final).to(device)

        optim_model=torch.optim.Adam(combined_model.parameters(), lr=lr_proposed)
        criterion = nn.CrossEntropyLoss()
        #----------------------------------------------------------------------------------------------------------------------------------


        #--------------------------my Unlearning Method------------------------------------------------------------------------------------
        starting_time = time.time()
        combined_model=modular_unlearning(combined_model, optim_model, criterion, device, beggining_epochs, intermediate_epochs, final_epochs, overture_epochs, final_thr, img_syn_loader, reduced_retainer[cycle])
        ending_time = time.time()
        unlearning_time.append(ending_time - starting_time)
        mia_score.append(measure_mia(combined_model, forgetter[cycle],test_loader, device))
        retain_acc.append(test(combined_model, retainer[cycle], device))
        forget_acc.append(test(combined_model, forgetter[cycle] , device))
        test_acc.append(test(combined_model, test_loader, device))


    stat_data = {
        'cycle': list(np.arange(num_cycles)+1),
        'mia_score': mia_score,
        'retain_acc': retain_acc,
        'forget_acc': forget_acc,
        'test_acc': test_acc,
        'unlearning_time': unlearning_time
    }

    df = pd.DataFrame(stat_data)

    file_path = os.path.join(result_directory_path,'modular_unlearning_stats.csv')
    df.to_csv(file_path, index=False)



    #--------------------------Retraining Method------------------------------------------------------------------------------------
    unlearning_time=[]
    mia_score=[]
    retain_acc=[]
    forget_acc=[]
    test_acc=[]

    for cycle in range(num_cycles):
        naive_net= CNN(channel, im_size[0], num_classes).to(device)
        starting_time = time.time()
        naive_net= retraining(naive_net, criterion, device, retrain_lr,retrain_epochs, retainer[cycle])
        ending_time = time.time()
        unlearning_time.append(ending_time - starting_time)

        mia_score.append(measure_mia(naive_net, forgetter[cycle],test_loader, device))
        retain_acc.append(test(naive_net, retainer[cycle], device))
        forget_acc.append(test(naive_net, forgetter[cycle], device))
        test_acc.append(test(naive_net, test_loader, device))


    stat_data = {
        'cycle': list(np.arange(num_cycles)+1),
        'mia_score': mia_score,
        'retain_acc': retain_acc,
        'forget_acc': forget_acc,
        'test_acc': test_acc,
        'unlearning_time': unlearning_time
    }


    df = pd.DataFrame(stat_data)

    file_path = os.path.join(result_directory_path,'retraining_unlearning_stats.csv')
    df.to_csv(file_path, index=False)


    #----------------------Catastrophic Forgetting Method---------------------------------------------------------------------------------
    unlearning_time=[]
    mia_score=[]
    retain_acc=[]
    forget_acc=[]
    test_acc=[]

    for cycle in range(num_cycles):
        naive_net= CNN(channel, im_size[0], num_classes).to(device)
        file_path = os.path.join(new_directory_path,'pretrained_net.pth')
        naive_net.load_state_dict(torch.load(file_path))
        starting_time = time.time()
        naive_net= retraining(naive_net, criterion, device, retrain_lr, retrain_epochs, retainer[cycle])
        ending_time = time.time()
        unlearning_time.append(ending_time - starting_time)

        mia_score.append(measure_mia(naive_net, forgetter[cycle],test_loader, device))
        retain_acc.append(test(naive_net, retainer[cycle], device))
        forget_acc.append(test(naive_net, forgetter[cycle], device))
        test_acc.append(test(naive_net, test_loader, device))


    stat_data = {
        'cycle': list(np.arange(num_cycles)+1),
        'mia_score': mia_score,
        'retain_acc': retain_acc,
        'forget_acc': forget_acc,
        'test_acc': test_acc,
        'unlearning_time': unlearning_time
    }


    df = pd.DataFrame(stat_data)

    file_path = os.path.join(result_directory_path,'catastrophic_unlearning_stats.csv')
    df.to_csv(file_path, index=False)



if __name__ == '__main__':
    main()