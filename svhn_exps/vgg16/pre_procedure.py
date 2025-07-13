
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
from model.model import *



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
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    training_epochs = 30
    batch_size = 256
    channel = 3
    im_size = (32,32)
    num_classes = 10
    lr_net=1e-4
    slicing_limit=4500
    #----------------------Hyperparameters---------------------------------



    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dst_train =  datasets.SVHN(root='./data', split='train', transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dst_train, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    dst_test = datasets.SVHN(root='./data', split='test', transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(dst_test, batch_size=batch_size,
                                            shuffle=False, num_workers=2)




    training_images=[]
    training_labels=[]

    for batch in train_loader:
        training_images.append(batch[0])
        training_labels.append(batch[1])


    training_images=torch.cat(training_images, dim=0)
    training_labels=torch.cat(training_labels, dim=0)

    training_images_curr_lab=[]
    training_labels_curr_lab=[]
    unique_labels_train=torch.unique(training_labels)
    for i in range(len(unique_labels_train)):
        curr_lab=unique_labels_train[i]
        # find indices of training images with label curr_lab
        indices_curr_lab=torch.where(training_labels==curr_lab)[0]
        # shuffle the indices
        indices_curr_lab=indices_curr_lab[torch.randperm(len(indices_curr_lab))]
        training_images_curr_lab.append(training_images[indices_curr_lab][:slicing_limit])
        training_labels_curr_lab.append(training_labels[indices_curr_lab][:slicing_limit])

    training_images=torch.cat(training_images_curr_lab, dim=0)
    training_labels=torch.cat(training_labels_curr_lab, dim=0)

    dst_train=TensorDatasett(training_images, training_labels)
    train_loader=torch.utils.data.DataLoader(dst_train, batch_size=batch_size, shuffle=True)


    test_images=[]
    test_labels=[]

    for batch in test_loader:
        test_images.append(batch[0])
        test_labels.append(batch[1])

    testing_images=torch.cat(test_images, dim=0)
    testing_labels=torch.cat(test_labels, dim=0)

    dst_test=TensorDatasett(testing_images, testing_labels)
    test_loader=torch.utils.data.DataLoader(dst_test, batch_size=batch_size, shuffle=True)


    #------------------Train the Net--------------------------------------------------------------------------------------------------------
    vgg16=modify_vgg16(channel, im_size[0], num_classes)
    net=Vgg16(vgg16=vgg16).to(device)
    for param in list(net.parameters()):
        param.requires_grad = True
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

        if epochy%2==0:
            print("Epoch: %d, Loss: %.3f"%(epochy, loss.item()))


    ending_time = time.time()
    training_time=ending_time - starting_time

    file_path = os.path.join(new_directory_path, 'pretrained_net.pth')
    torch.save(net.state_dict(), file_path)

    train_acc=test(net, train_loader, device)
    test_acc=test(net, test_loader, device)
    print("Train accuracy: %.2f"%train_acc)
    print("Test accuracy: %.2f"%test_acc)

    stat_data = {
        'Training Time': [training_time],
        'Train Accuracy': [train_acc],
        'Test Accuracy': [test_acc]
    }

    df = pd.DataFrame(stat_data)
    
    file_path = os.path.join(result_directory_path, 'stat_vgg16_cifar10_'+str(training_epochs)+'epochs.csv')
    df.to_csv(file_path, index=False)

    file_path = os.path.join(new_directory_path,'train_dataset.pth')
    torch.save(dst_train, file_path)
    file_path = os.path.join(new_directory_path,'test_dataset.pth')
    torch.save(dst_test, file_path)




if __name__ == '__main__':
    main()

