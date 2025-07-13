

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
from auxil.auxils import *
from model.model_resnet import *




def offline_trainer(new_directory_path,net,img_real_data_loader, img_syn_loader, retain_loader, forget_loader, device, offline_condensation_iterations, final_model_epochs, databank_model_epochs, lr_final, lr_databank):
    ref_net=copy.deepcopy(net)
    for param in list(ref_net.parameters()):
        param.requires_grad = False

    beggining=Beginning(net).to(device)

    intermediate=Intermediate(net).to(device)
 
    databank=Databank(beggining=beggining, intermediate=intermediate).to(device)
    #---------------------------Offline Beethoven-------------------------------------------------
    for it in range(offline_condensation_iterations):

        for param in list(databank.parameters()):
            param.requires_grad = False


        pseudo_ref_net=copy.deepcopy(ref_net)
        final=Final(pseudo_ref_net).to(device)

        for param in list(final.parameters()):
            param.requires_grad = True

        criterion = nn.CrossEntropyLoss()
        optimizer_final=torch.optim.Adam(final.parameters(), lr=lr_final)
        

        # training of final part
        for ep in range(final_model_epochs):
            run_loss=0.0
            for batch in img_syn_loader:
                img_syn_buffer, label_img_syn_buffer= batch
                img_syn_buffer=img_syn_buffer.to(device)
                label_img_syn_buffer=label_img_syn_buffer.to(device)
                decoded_img_syn_buffer=databank(img_syn_buffer)
                output=final(decoded_img_syn_buffer)
                loss=criterion(output, label_img_syn_buffer)
                optimizer_final.zero_grad()
                loss.backward()
                optimizer_final.step()
                run_loss+=loss.item()



        del optimizer_final
        del loss

        # make final non-trainable
        for param in list(final.parameters()):
            param.requires_grad = False

        # make databank's beggining non-trainable
        for param in list(databank.beggining.parameters()):
            param.requires_grad = False

        # make databank's intermediate trainable
        for param in list(databank.intermediate.parameters()):
            param.requires_grad = True


        optimizer_databank=torch.optim.Adam(databank.parameters(), lr=lr_databank)

        # training the databank's intermediate part
        for ep in range(databank_model_epochs):
            run_loss=0.0
            for batch in img_real_data_loader:
                img_real_buffer, label_img_real_buffer= batch
                img_real_buffer=img_real_buffer.to(device)
                label_img_real_buffer=label_img_real_buffer.to(device)
                approx_img_real_buffer=databank(img_real_buffer)
                output=final(approx_img_real_buffer)
                loss=criterion(output, label_img_real_buffer)
                optimizer_databank.zero_grad()
                loss.backward()
                optimizer_databank.step()
                run_loss+=loss.item()

            if ep==0:
                print("Loss associated with databank (it's intermediate): %.3f"%(run_loss/len(img_real_data_loader)))

            # if ep==databank_model_epochs-1:
            #     combined_model=CombinedModel(databank=databank, final=final).to(device)
            #     with torch.no_grad():
            #         combined_retrain_acc=test(combined_model, retain_loader, device)
            #         combined_forget_acc=test(combined_model, forget_loader, device)

            #         print("Combined model's accuracy on retain set: %.2f"%combined_retrain_acc)
            #         print("Combined model's accuracy on forget set: %.2f"%combined_forget_acc)
                    


        # finally make databank's intermediate non-trainable
        for param in list(databank.intermediate.parameters()):
            param.requires_grad = False


    print("\nSaving the databank")
    file_path = os.path.join(new_directory_path,'databank.pth')
    torch.save(databank.state_dict(), file_path)


    print("\n Saving the final model\n\n")
    file_path = os.path.join(new_directory_path,'final.pth')
    torch.save(final.state_dict(), file_path)
