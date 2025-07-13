import warnings
warnings.filterwarnings('ignore')
import os
import torch
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

# Assuming you have the code you provided above

# Normalize images to [0, 1]
def normalize_images(images):
    min_val = images.min()
    max_val = images.max()
    normalized_images = (images - min_val) / (max_val - min_val)
    return normalized_images

# Create a function to plot images as a matrix
def plot_images_matrix(images_dict, class_names, figsize, filename):
    num_classes = len(class_names)
    num_rows = max(len(images_dict[class_name]) for class_name in class_names)
    num_cols = num_classes
    
    plt.figure(figsize=figsize)
    
    for i, class_name in enumerate(class_names):
        images = images_dict[class_name]
        num_images = len(images)
        
        for j, image in enumerate(images):
            plt.subplot(num_rows, num_cols, j * num_classes + i + 1)
            normalized_image = normalize_images(np.transpose(image, (1, 2, 0)))
            plt.imshow(normalized_image)
            plt.axis('off')
            if j == 0:
                plt.title(class_name)
    
    plt.tight_layout()

    # Save the figure to a file
    plt.savefig(filename)
    
    # Close the figure to free up memory
    plt.close()


def class_sampler(train_loader, samp_num_classes):

    # find unique classes in train_loader
    unique_classes = []
    for batch in train_loader:
        _,label = batch
        unique_classes.append(label)

    unique_classes=torch.cat(unique_classes).cpu().numpy()
    unique_classes = np.unique(unique_classes)

    # select random classes from unique classes
    random_classes = random.sample(list(unique_classes), samp_num_classes)
    return random_classes



def Attack(mynet, img_shape, num_classes, target_label, lr_img, mi_epochs,device):
    mynet.to(device)
    for param in list(mynet.parameters()):
        param.requires_grad = False

    syn_image = torch.zeros(img_shape).to(device)
    syn_image.requires_grad = True
    optim_img=torch.optim.Adam([syn_image,], lr=lr_img)

    for i in range(mi_epochs):
        out = mynet.forward(syn_image)
        out = out.reshape(1, num_classes)
        target_class = torch.tensor([target_label]).to(device)
        cost = nn.CrossEntropyLoss()(out, target_class.long())
        optim_img.zero_grad()
        cost.backward()
        optim_img.step()
    inverted_image = syn_image.detach()
    return inverted_image
    



def basic_model_inversion(directory, net, img_shape, num_classes, random_classes, lr_img, mi_epochs, case, device):
    
    images_dict = {class_name: [] for class_name in random_classes}
    ext_img_shape = (1,) + img_shape
    for each_class in random_classes:
        inverted_img=Attack(net, ext_img_shape, num_classes, each_class, lr_img, mi_epochs, device).cpu().squeeze(0).numpy()
        images_dict[each_class].append(inverted_img)


    file_path = os.path.join(directory,'MIAttack_'+str(case)+'.png')
    plot_images_matrix(images_dict, random_classes, figsize=(15, 15),filename=file_path)





    