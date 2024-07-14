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
from torchvision.models import resnet18
import pickle
import torchvision.models as models

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
from auxil.fisher_forget import *
from auxil.ntk_scrubbing import *





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

    overture_epochs=10
    beggining_epochs=2
    final_epochs=50
    intermediate_epochs= 5
    final_thr=2   # intended for blocking the final training in overture, 
                # from the end of overture epochs--> improves retain acc while preserving forget accuracy

    threshold = 0.5  # Choose an appropriate threshold for binarizing the Fisher Information
    lambd=0.1   #noise addition magnitude


    retrain_lr=1e-3
    retrain_epochs=30


    #------------------------------------------------------------------------


    #----------------------------Loading stuff------------------------------------------------------------------------
    net=ConvAutoencoder(input_channels=channel, im_size=im_size[0]).to(device)
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

    net_for_ff=copy.deepcopy(net)
    net_for_ntk=copy.deepcopy(net)
    #--------------------------Initializing my Unlearning Method--------------------------------------------------------------------------- 
    ref_net=copy.deepcopy(net)
    for param in list(ref_net.parameters()):
        param.requires_grad = False

    beggining=Beginning(ref_net).to(device)

    intermediate=Intermediate(ref_net).to(device)
 
    data_bank=Databank(beggining=beggining, intermediate=intermediate).to(device)
    file_path = os.path.join(new_directory_path,'databank.pth')
    data_bank.load_state_dict(torch.load(file_path))

    conv_cl=ConvClassifier(input_channels=channel, im_size=im_size[0], num_classes=num_classes)

    final=Final(conv_cl).to(device)
    file_path = os.path.join(new_directory_path,'final.pth')
    final.load_state_dict(torch.load(file_path))

    combined_model=CombinedModel(databank=data_bank, final=final).to(device)
    ref_combined_model=copy.deepcopy(combined_model)

    optim_model=torch.optim.Adam(combined_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    #----------------------------------------------------------------------------------------------------------------------------------

    #--------------------------Reference model------------------------------------------------------------------------------------
    mia_score=measure_mia(ref_combined_model, forget_loader,test_loader, device)
    retain_acc=test(ref_combined_model, retain_loader, device)
    forget_acc=test(ref_combined_model, forget_loader, device)
    test_acc=test(ref_combined_model, test_loader, device)

    print('\nReference Model Stats: ')
    print('======================================')
    print('mia_score: ', mia_score)
    print('retain_acc: ', retain_acc)
    print('forget_acc: ', forget_acc)
    print('test_acc: ', test_acc)
    print('======================================')

    # save as csv file in /result
    df = pd.DataFrame({'mia_score': [mia_score], 'retain_acc': [retain_acc], 'forget_acc': [forget_acc], 'test_acc': [test_acc]})
    df.to_csv(os.path.join(result_directory_path,'reference_model.csv'), index=False)


    #----------------------Fischer-Forgetting Method---------------------------------------------------------------------------------
    starting_time = time.time()
    forgot_net=fisher_forgetting(net_for_ff, retain_loader, forget_loader, device=device, alpha=1e-6, seed=42)
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
    df.to_csv(os.path.join(result_directory_path,'fisher_forgetting_stats.csv'), index=False)

    
    #----------------------NTK-Scrubbing Method---------------------------------------------------------------------------------
    starting_time = time.time()
    forgot_net = ntk_scrubbing(net_for_ntk, retain_loader, forget_loader, device=device, num_classes=num_classes, weight_decay=1e-1, beta=1)
    ending_time = time.time()
    unlearning_time=ending_time - starting_time

    mia_score = measure_mia(forgot_net, forget_loader, test_loader, device)
    retain_acc = test(forgot_net, retain_loader, device)
    forget_acc = test(forgot_net, forget_loader, device)

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
    df.to_csv(os.path.join(result_directory_path,'ntk_scrubbing_stats.csv'), index=False)
    


    #--------------------------my Unlearning Method------------------------------------------------------------------------------------
    starting_time = time.time()
    combined_model=modular_unlearning(combined_model, optim_model, criterion, device, beggining_epochs, intermediate_epochs, final_epochs, overture_epochs, final_thr, img_syn_loader, reduced_retain_loader)
    ending_time = time.time()
    unlearning_time=ending_time - starting_time
    mia_score=measure_mia(combined_model, forget_loader,test_loader, device)
    retain_acc=test(combined_model, retain_loader, device)
    forget_acc=test(combined_model, forget_loader, device)
    test_acc=test(combined_model, test_loader, device)

    print('\nModular Unlearning Stats: ')
    print('======================================')
    print('mia_score: ', mia_score)
    print('retain_acc: ', retain_acc)
    print('forget_acc: ', forget_acc)
    print('test_acc: ', test_acc)
    print('unlearning_time: ', unlearning_time)
    print('======================================')

    # save as csv file in /result
    df = pd.DataFrame({'mia_score': [mia_score], 'retain_acc': [retain_acc], 'forget_acc': [forget_acc], 'test_acc': [test_acc], 'unlearning_time': [unlearning_time]})
    df.to_csv(os.path.join(result_directory_path,'modular_unlearning.csv'), index=False)


    vgg16 =  modify_vgg16(channel, im_size[0], num_classes).to(device)
    combined_model.final=final
    for param in list(combined_model.databank.parameters()):
        param.requires_grad = False
    for param in list(combined_model.final.parameters()):
        param.requires_grad = True
    optim_model=torch.optim.Adam(combined_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    starting_time = time.time()

    # Teacher training before starting the dataset distillation process
    for epochy in range(30):
        for batch in img_syn_loader:
            data, label = batch[0].to(device), batch[1].to(device)
            output = combined_model(data) 
            loss = nn.CrossEntropyLoss()(output, label)
            optim_model.zero_grad()
            loss.backward()
            optim_model.step()
    ending_time = time.time()
    training_time=ending_time - starting_time

    mia_score=measure_mia(combined_model, forget_loader,test_loader, device)
    test_acc=test(combined_model, test_loader, device)
    retain_acc=test(combined_model, retain_loader, device)
    forget_acc=test(combined_model, forget_loader, device)

    print('\nRe-Condensation Training Stats: ')
    print('======================================')
    print('training_time: ', training_time)
    print('test_acc: ', test_acc)
    print('retain_acc: ', retain_acc)
    print('forget_acc: ', forget_acc)
    print('mia_score: ', mia_score)
    print('======================================')

    # save as csv file in /result
    df = pd.DataFrame({'mia_score': [mia_score], 'retain_acc': [retain_acc], 'forget_acc': [forget_acc], 'test_acc': [test_acc], 'training_time': [training_time]})
    df.to_csv(os.path.join(result_directory_path,'recondensation_training.csv'), index=False)


    vgg16 = models.vgg16(pretrained=True)
    
    # Modify the first convolutional layer to accept custom input shape
    vgg16.features[0] = nn.Conv2d(channel, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    
    # Determine the classifier input size based on the flattened output size
    classifier_input_size = 512
    
    # Adjusting the input features of the classifier
    vgg16.classifier[0] = nn.Linear(classifier_input_size, 4096)
    
    
    vgg16.classifier[6] = nn.Linear(4096, num_classes)
    vgg16=Vgg16(vgg16).to(device)

    for param in list(vgg16.parameters()):
        param.requires_grad = True
    optim_model=torch.optim.Adam(vgg16.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    starting_time = time.time()

    # Teacher training before starting the dataset distillation process
    for epochy in range(30):
        for batch in retain_loader:
            data, label = batch[0].to(device), batch[1].to(device)
            output = vgg16(data) 
            loss = nn.CrossEntropyLoss()(output, label)
            optim_model.zero_grad()
            loss.backward()
            optim_model.step()
        
    ending_time = time.time()
    training_time=ending_time - starting_time

    mia_score=measure_mia(vgg16, forget_loader,test_loader, device)
    test_acc=test(vgg16, test_loader, device)
    retain_acc=test(vgg16, retain_loader, device)
    forget_acc=test(vgg16, forget_loader, device)

    print('\nRe-Training Stats: ')
    print('======================================')
    print('training_time: ', training_time)
    print('test_acc: ', test_acc)
    print('retain_acc: ', retain_acc)
    print('forget_acc: ', forget_acc)
    print('mia_score: ', mia_score)
    print('======================================')

    # save as csv file in /result
    df = pd.DataFrame({'mia_score': [mia_score], 'retain_acc': [retain_acc], 'forget_acc': [forget_acc], 'test_acc': [test_acc], 'training_time': [training_time]})
    df.to_csv(os.path.join(result_directory_path,'retraining.csv'), index=False)




if __name__ == '__main__':
    main()