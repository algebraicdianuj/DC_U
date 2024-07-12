
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
from torchvision.models import resnet18
from auxil.auxils import *
from model.model_vgg import *
from modular_unlearn.offline_training_vgg import *
from modular_unlearn.ds_condensation_imrpvDM import *
from modular_unlearn.modular_forgetting import *


def test(model, data_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


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
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    training_epochs = 30
    batch_size = 256
    channel = 3
    im_size = (32,32)
    num_classes = 10
    lr_net=1e-3
    #----------------------Hyperparameters---------------------------------



    file_path = os.path.join(new_directory_path,'train_dataset.pth')
    dst_train=torch.load(file_path)
    file_path = os.path.join(new_directory_path,'test_dataset.pth')
    dst_test=torch.load(file_path)
    train_loader=torch.utils.data.DataLoader(dst_train, batch_size=batch_size, shuffle=True)
    test_loader=torch.utils.data.DataLoader(dst_test, batch_size=batch_size, shuffle=True)


    #------------------Train the Net--------------------------------------------------------------------------------------------------------
    vgg16=modify_vgg16(channel, im_size[0], num_classes)
    net=Vgg16(vgg16=vgg16).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(net.parameters(), lr=lr_net)

    starting_time = time.time()

    # Teacher training before starting the dataset distillation process
    for epochy in range(training_epochs):
        for batch in train_loader:
            data, target = batch[0].to(device), batch[1].to(device)
            output = net(data) 
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    ending_time = time.time()
    training_time=ending_time - starting_time

    file_path = os.path.join(new_directory_path, 'pretrained_net.pth')
    torch.save(net.state_dict(), file_path)

    train_acc=test(net, train_loader, device)
    test_acc=test(net, test_loader, device)

    stat_data = {
        'Training Time': [training_time],
        'Train Accuracy': [train_acc],
        'Test Accuracy': [test_acc]
    }

    df = pd.DataFrame(stat_data)
    
    file_path = os.path.join(result_directory_path, 'stat_vgg16_cifar10_'+str(training_epochs)+'epochs.csv')
    df.to_csv(file_path, index=False)


    #----------------------Hyperparameters---------------------------------
    batch_size = 256
    batch_size_bucket=128
    batch_syn=32
    channel = 3
    im_size = (32, 32)

    num_classes = 10
    batch_real = 5000
    split_ratio = 0.1   # forget-retain split ratio
    n_subclasses= 450   # K-means "K"
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



if __name__ == '__main__':
    main()

