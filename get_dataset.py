import warnings
warnings.filterwarnings('ignore')

import os
import torch
import torch.nn as nn
import numpy as np
import time
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import pandas as pd

import argparse
from utils.utils import TensorDatasett


def main(args):

    directory_name = 'reservoir'
    current_path = os.getcwd()
    data_storage = os.path.join(current_path, directory_name)
    
    if not os.path.exists(data_storage):
        os.makedirs(data_storage)
        print(f"Directory '{directory_name}' created in the current working directory.")
    else:
        print(f"Directory '{directory_name}' already exists in the current working directory.")
    
    dat_dir = 'result'
    result_directory_path = os.path.join(current_path, dat_dir)
    
    if not os.path.exists(result_directory_path):
        os.makedirs(result_directory_path)
        print(f"Directory '{dat_dir}' created in the current working directory.")
    else:
        print(f"Directory '{dat_dir}' already exists in the current working directory.")
    

    if args.choice == 'cifar10':
        # ---------------------- Hyperparameters --------------------------------
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        im_size = (32,32)
        batch_size = 32
        slicing_limit = 5000
        # -----------------------------------------------------------------------

        transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        dst_train = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )

        dst_test = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

    elif args.choice == 'svhn':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        batch_size = 32
        slicing_limit = 4500
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.SVHN('./data', split='train', download=True, transform=transform)  # no augmentation
        dst_test = datasets.SVHN('./data', split='test', download=True, transform=transform)



    elif args.choice == 'cinic10':
        cinic_directory = './data'
        mean = [0.478, 0.472, 0.430]
        std = [0.242, 0.238, 0.258]

        im_size = (32, 32)
        batch_size = 32
        slicing_limit = 9000
        
        transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        dst_train = datasets.ImageFolder(os.path.join(cinic_directory, 'train'), transform=transform)
        dst_test = datasets.ImageFolder(os.path.join(cinic_directory, 'test'), transform=transform)
            
    elif args.choice == 'cifar100':
        # ---------------------- Hyperparameters --------------------------------
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        im_size = (32,32)
        batch_size = 256
        slicing_limit = 500
        # -----------------------------------------------------------------------

        transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        dst_train = datasets.CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )

        dst_test = datasets.CIFAR100(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

    else:
        raise ValueError("Invalid dataset choice. Currently choose from 'cifar10', 'svhn', 'cinic10' or 'cifar100'.")
    

    torch.save(mean, os.path.join(data_storage, 'means.pt'))
    torch.save(std, os.path.join(data_storage, 'stds.pt'))
    torch.save(im_size, os.path.join(data_storage, 'im_size.pt'))
    

    train_loader = torch.utils.data.DataLoader(dst_train, batch_size=batch_size,
                                               shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dst_test, batch_size=batch_size,
                                              shuffle=False, num_workers=0)
    
    if args.choice == 'svhn':
        unique_labels_train = torch.arange(10)

    else:
        train_labels_all = torch.tensor(dst_train.targets) 
        unique_labels_train = train_labels_all.unique()
    num_classes = len(unique_labels_train)
    torch.save(num_classes, os.path.join(data_storage, 'num_classes.pt'))
    
    class_counts = {int(cls.item()): 0 for cls in unique_labels_train}
    
    final_train_images = []
    final_train_labels = []
    
    for batch_imgs, batch_labels in train_loader:
        for img, lbl in zip(batch_imgs, batch_labels):
            cls = int(lbl.item())
            if class_counts[cls] < slicing_limit:
                final_train_images.append(img.unsqueeze(0))
                final_train_labels.append(lbl.unsqueeze(0))
                class_counts[cls] += 1


    final_train_images = torch.cat(final_train_images, dim=0)
    final_train_labels = torch.cat(final_train_labels, dim=0)

    dst_train_sliced = TensorDatasett(final_train_images, final_train_labels)


    test_images = []
    test_labels = []

    for batch_imgs, batch_labels in test_loader:
        test_images.append(batch_imgs)
        test_labels.append(batch_labels)

    final_test_images = torch.cat(test_images, dim=0)
    final_test_labels = torch.cat(test_labels, dim=0)

    dst_test_sliced = TensorDatasett(final_test_images, final_test_labels)


    dst_train=dst_train_sliced
    dst_test=dst_test_sliced

    torch.save(dst_train, os.path.join(data_storage, 'train_dataset.pth'))
    torch.save(dst_test, os.path.join(data_storage, 'test_dataset.pth'))

    print("Done creating and saving datasets.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
  
    parser.add_argument('--choice',
                        type=str,
                        default='cifar10',
                        choices=['cifar10', 'cfar100', 'svhn', 'cinic10'],
                        help='Dataset choice: cifar10 or cinic10')
    args = parser.parse_args()
    main(args)
