
import warnings
warnings.filterwarnings('ignore')
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
import sys
from auxil.auxils import *




def Average(ref_imgs_all, pretrained, num_epochs=100,device=torch.device('cpu')):

    ref_imgs_all=ref_imgs_all.to(device)
    
    weighted_avg_module = WeightedAverage(num_batches=ref_imgs_all.shape[0]).to(device)
    optim_weighted_avg = torch.optim.Adam(weighted_avg_module.parameters(), lr=1e-3)

    ref_features= pretrained.feature(ref_imgs_all).detach()

    for ep in range(num_epochs):
        fused_img= weighted_avg_module(ref_imgs_all)
        fused_img_features= pretrained.feature(fused_img)
        loss=torch.sum((torch.mean(ref_features, dim=0) - torch.mean(fused_img_features, dim=0))**2)
        optim_weighted_avg.zero_grad()
        loss.backward()
        optim_weighted_avg.step()

    averaged_img=weighted_avg_module(ref_imgs_all).detach()

    return averaged_img





def improv_DM_condensation(new_lab_train, train_images, train_labels, net, condensation_epochs, n_classes,n_subclasses, device):
    #-----------------------------Dataset Condensation and Offline data processing--------------------------------------------------------------
    inverted_IMG=[]
    inverted_LABEL=[]
    indices_train_wrt_bucket=[]

    bucket_labbies=torch.unique(new_lab_train).tolist()

    for idx in bucket_labbies:

        indices_idx = torch.where(new_lab_train.to(device)==idx)[0]

        indices_train_wrt_bucket.append(indices_idx.cpu())

        ref_imgs_all = train_images[indices_idx.cpu()]
        ref_labs_all= train_labels[indices_idx.cpu()]

        inverted_image = Average(ref_imgs_all, pretrained=net, num_epochs=condensation_epochs, device=device)

        inverted_IMG.append(inverted_image)
        inverted_LABEL.append(ref_labs_all[0])

        # print percentage of idx covered on same line
        print('\r','Condensation Progress: ', (idx+1)*100/(n_classes*n_subclasses), '%', end='')

    inverted_IMG=torch.cat(inverted_IMG, dim=0).cpu()
    inverted_LABEL=torch.tensor(inverted_LABEL).cpu()


    return inverted_IMG, inverted_LABEL, indices_train_wrt_bucket, bucket_labbies
 





def offline_data_processing(inverted_IMG, inverted_LABEL, indices_train_wrt_bucket, forget_indices, retain_indices, bucket_labbies, img_real_data_dataset,train_images,train_labels,forget_loader,retain_loader,device):
    
    indices_collector=[]
    not_safe_zones=[]    # foret has been found here so corresponding condensed is of no use (and i have to residue it)

    # in bucketting system, find the retain indices where forget is found
    for i in range(len(forget_indices)):
        for j in range(len(indices_train_wrt_bucket)):

            if forget_indices[i] in indices_train_wrt_bucket[j]:

                not_safe_zones.append(j)

                #find the indexes of the indices_train_wrt_bucket[j] that are not equal to forget_labels[i]
                false_indices = [idx for idx, val in enumerate(indices_train_wrt_bucket[j]) if val != forget_indices[i]]

                # only select those false_indices that lie in retain_indices
                false_indices_subset = [x for x in false_indices if indices_train_wrt_bucket[j][x] in retain_indices]


                if len(false_indices_subset)!=0:
                    indices_collector.append(indices_train_wrt_bucket[j][false_indices_subset].tolist())

                break

        print('\r','Residual Collection Progress: ', i*100/(len(forget_indices)), '%', end='')

    #----------------------------------------------------------------------------------------------------------------------------------------------


    print("\n\nSize of each bucket: ", int(len(img_real_data_dataset)/len(bucket_labbies)))

    not_safe_zones=torch.tensor(not_safe_zones)
    not_safe_zones=torch.unique(not_safe_zones).tolist()
    print("\n\nFaulty Buckets: ", len(not_safe_zones), '/', len(bucket_labbies))

    # convert indices_collector to a flat list 
    possible_retain_sols = [item for sublist in indices_collector for item in sublist]

    retain_sols=torch.unique(torch.tensor(possible_retain_sols)).tolist()


    residual_retain_imgs=train_images[retain_sols]
    residual_retain_labels=train_labels[retain_sols]


    print("\n\nResidual Retain Images (in bucketting system where retain was found alongside with forget): ", len(residual_retain_imgs))


    total_retain_imgs=[]
    total_retain_labs=[]

    total_retain_imgs.append(residual_retain_imgs)
    total_retain_labs.append(residual_retain_labels)


    # safe zone is indices in indices_train_wrt_bucket that are not in not_safe_zones
    safe_zone=[]
    for i in range(len(indices_train_wrt_bucket)):
        if i not in not_safe_zones:
            safe_zone.append(i)


    if len(safe_zone)!=0:
        safe_zone=torch.tensor(safe_zone)
        safe_zone=torch.unique(safe_zone)

        condensed_retain_imgs=inverted_IMG[safe_zone]
        condensed_retain_labels=inverted_LABEL[safe_zone]

        total_retain_imgs.append(condensed_retain_imgs)
        total_retain_labs.append(condensed_retain_labels)

        print("Size of usable condensed images: ", len(condensed_retain_imgs), '/', len(inverted_IMG), 'buckets')


    total_retain_imgs=torch.cat(total_retain_imgs, dim=0)
    total_retain_labs=torch.cat(total_retain_labs, dim=0)

    print("---------------------------------------------------")
    print(">> Total size of Reduced Retain Set: ", len(total_retain_imgs))
    print(">> Reference size of naive retain loader:", len(retain_loader.dataset))
    print(">> Retain Compression Ratio (>=1): %.2f"%(len(retain_loader.dataset)/len(total_retain_imgs)))
    print("---------------------------------------------------")


    return total_retain_imgs,total_retain_labs
#---------------------------------------------------------------------------------------------------------------------------------------------