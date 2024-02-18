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
from auxil.auxils import *
from model.model import *
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



    #----------------------Hyperparameters---------------------------------
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    channel = 3
    im_size = (32, 32)
    num_classes = 10
    batch_real = 5000
    split_ratio = 0.1   # forget-retain split ratio
    n_subclasses= 450   # K-means "K"
    seeder=42
    ipc=10
    choice= 'arbitrary_uniform'    # 'arbitrary_uniform' or 'classwise' or 'arbitrary_partial' 'mia_protection'
    forgetfull_class=0     #  if choice=='classwise' then this is the class to forget
    forgetfull_class_list = [1,2,3]    # if choice=='arbitrary_partial' then this is the list of classes to forget
    #----------------------Hyperparameters---------------------------------


    #---------------------------------------Starting code---------------------------------------
    file_path = os.path.join(new_directory_path,'train_dataset.pth')
    dst_train=torch.load(file_path)
    file_path = os.path.join(new_directory_path,'test_dataset.pth')
    dst_test=torch.load(file_path)
    net=vitl(im_size=im_size[0], num_classes=num_classes).to(device)
    file_path = os.path.join(new_directory_path,'pretrained_net.pth')
    net.load_state_dict(torch.load(file_path))
    test_loader = torch.utils.data.DataLoader(dst_test, batch_size=batch_size,
                                                shuffle=False, num_workers=2)

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]

    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to(device)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=device)

    def get_images(c, n): # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    IMG_real=[]
    LAB_real=[]
    for c in range(num_classes):
        IMG_real.append(get_images(c, batch_real))
        LAB_real.append(torch.ones(batch_real, dtype=torch.long, device=device)*c)

    IMG_real=torch.cat(IMG_real, dim=0)
    LAB_real=torch.cat(LAB_real, dim=0)


    img_real_list=[]
    lab_real_list=[]
    img_real_sampled_list=[]
    lab_real_sampled_list=[]


    for c in range(num_classes):
        img_real = IMG_real[c * batch_real: (c + 1) * batch_real].clone().detach()
        lab_real = LAB_real[c * batch_real: (c + 1) * batch_real].clone().detach()
        img_real_list.append(img_real)
        lab_real_list.append(lab_real)

        sampled_img_real = get_images_from_testloader(ipc, c,test_loader)
        img_real_sampled_list.append(sampled_img_real)
        lab_real_sampled_list.append(torch.ones(sampled_img_real.shape[0], dtype=torch.long, device=device)*c)


    img_real_data=torch.cat(img_real_list, dim=0)
    lab_real_data=torch.cat(lab_real_list, dim=0)
    img_real_sampled_data=torch.cat(img_real_sampled_list, dim=0)
    lab_real_sampled_data=torch.cat(lab_real_sampled_list, dim=0)


    img_real_data_dataset=TensorDatasett(img_real_data.clone().detach().cpu(), lab_real_data.clone().detach().cpu())


    img_syn_dataset=TensorDatasett(img_real_sampled_data.clone().detach().cpu(), lab_real_sampled_data.clone().detach().cpu())


    # Assuming img_real_data_dataset is predefined
    dataset_size = len(img_real_data_dataset)

    train_images=img_real_data_dataset.images
    train_labels=img_real_data_dataset.labels


    new_lab_train, original_labels_dict = create_sub_classes(train_images, train_labels, model=net, num_classes=num_classes, sub_divisions=n_subclasses, device=device)


    if choice=='arbitrary_uniform':

        indices = list(range(dataset_size))

        # shuffling the indices
        torch.manual_seed(seeder)  # for reproducibility
        indices = torch.randperm(dataset_size)


        new_lab_train= new_lab_train[indices]
        train_images = train_images[indices]
        train_labels = train_labels[indices]


        # Define split ratio and sizes
        split = int(split_ratio * dataset_size)

        # Split indices into two parts
        forget_indices = indices[:split]
        retain_indices = indices[split:]


        forget_images=train_images[forget_indices]
        forget_labels=train_labels[forget_indices]

        retain_images=train_images[retain_indices]
        retain_labels=train_labels[retain_indices]


    elif choice=='classwise':

        indices = list(range(dataset_size))

    
        # shuffling the indices
        torch.manual_seed(seeder)  # for reproducibility
        indices = torch.randperm(dataset_size)

        new_lab_train= new_lab_train[indices]
        train_images = train_images[indices]
        train_labels = train_labels[indices]


        # Find indices of images that belong to class 'c'
        forget_indices = [i for i, label in enumerate(train_labels) if label == forgetfull_class]

        # Find indices of images that do not belong to class 'c'
        retain_indices = [i for i, label in enumerate(train_labels) if label != forgetfull_class]

        # Now, update the dataset and images/labels
        forget_images = train_images[forget_indices]
        forget_labels = train_labels[forget_indices]

        retain_images = train_images[retain_indices]
        retain_labels = train_labels[retain_indices]


    elif choice=='arbitrary_partial':

        indices = list(range(dataset_size))

        # shuffling the indices
        torch.manual_seed(seeder)  # for reproducibility
        indices = torch.randperm(dataset_size)

        new_lab_train= new_lab_train[indices]
        train_images = train_images[indices]
        train_labels = train_labels[indices]


        class_indices = [i for i, label in enumerate(train_labels) if label in forgetfull_class_list]


        class_indices = torch.tensor(class_indices)[torch.randperm(len(class_indices))]

        num_forget = int(split_ratio * len(class_indices))

        forget_indices = class_indices[:num_forget]

        other_indices = [i for i, label in enumerate(train_labels) if label not in forgetfull_class_list]

        retain_indices = torch.cat((torch.tensor(other_indices), class_indices[num_forget:]))

        forget_indices = forget_indices.tolist()
        retain_indices = retain_indices.tolist()

        # shuffle retain_indices
        random.shuffle(retain_indices)

        # Now, update the dataset and images/labels
        forget_images = train_images[forget_indices]
        forget_labels = train_labels[forget_indices]

        retain_images = train_images[retain_indices]
        retain_labels = train_labels[retain_indices]


    elif choice=='mia_protection':

        indices = list(range(dataset_size))

        # shuffling the indices
        torch.manual_seed(seeder)  # for reproducibility
        indices = torch.randperm(dataset_size)


        new_lab_train= new_lab_train[indices]
        train_images = train_images[indices]
        train_labels = train_labels[indices]

        forget_indices, retain_indices, forget_images, forget_labels, retain_images, retain_labels = get_overfitting_samples(net, nn.CrossEntropyLoss(), train_images, train_labels, indices, device)


    else:
        raise ValueError('Invalid choice of choice')


    print('Ratio of Forget to Retain: ', len(forget_indices)/len(retain_indices))

    img_real_data_dataset=TensorDatasett(train_images, train_labels)
    forget_set_real=TensorDatasett(forget_images, forget_labels)
    retain_set_real=TensorDatasett(retain_images, retain_labels)


    file_path = os.path.join(new_directory_path,'forget_set.pth')
    torch.save(forget_set_real, file_path)
    file_path = os.path.join(new_directory_path,'retain_set.pth')
    torch.save(retain_set_real, file_path)
    file_path = os.path.join(new_directory_path,'test_set.pth')
    torch.save(dst_test, file_path)
    file_path = os.path.join(new_directory_path,'indices.pth')
    torch.save(indices, file_path)
    file_path = os.path.join(new_directory_path,'forget_indices.pth')
    torch.save(forget_indices, file_path)
    file_path = os.path.join(new_directory_path,'retain_indices.pth')
    torch.save(retain_indices, file_path)
    file_path = os.path.join(new_directory_path,'clustered_label_train.pth')
    torch.save(new_lab_train, file_path)
    file_path = os.path.join(new_directory_path,'image_train.pth')
    torch.save(train_images, file_path)
    file_path = os.path.join(new_directory_path,'label_train.pth')
    torch.save(train_labels, file_path)
    file_path = os.path.join(new_directory_path,'syn_set.pth')
    torch.save(img_syn_dataset, file_path)
    file_path = os.path.join(new_directory_path,'Klabels_labels_dict.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(original_labels_dict, file)


if __name__ == '__main__':
    main()


