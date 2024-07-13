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
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet18
import pickle

from modular_unlearn.offline_training import *
from modular_unlearn.ds_condensation_imrpvDM import *
from modular_unlearn.modular_forgetting import *
from modular_unlearn.unlearning_metric import *
from modular_unlearn.overfitting_metric import *
from auxil.auxils import *
from model.model import *
from auxil.retrain import *
from auxil.distillation import *
from auxil.sparisification import *
from auxil.bad_distillation import *
from auxil.dp_train import *
from auxil.mia_forget_logit import *
from auxil.trivial_mi import *
from auxil.innovative_mi_singleimg import *
from auxil.innovative_mi_multiimg import *
from auxil.mia_whole import *



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
    intermediate_epochs= 4
    final_thr=2   # intended for blocking the final training in overture, 
                # from the end of overture epochs--> improves retain acc while preserving forget accuracy

    threshold = 0.5  # Choose an appropriate threshold for binarizing the Fisher Information
    lambd=0.1   #noise addition magnitude


    retrain_lr=1e-3
    retrain_epochs=30
    MAX_GRAD_NORM = 20.0
    EPSILON = 10.0
    DELTA = 1e-1
    MULTIPLIER=1.0

    #------------------------------------------------------------------------


    #----------------------------Loading stuff------------------------------------------------------------------------
    vgg16=modify_vgg16(channel, im_size[0], num_classes)
    net=Vgg16(vgg16=vgg16).to(device)
    file_path = os.path.join(new_directory_path,'pretrained_net.pth')
    net.load_state_dict(torch.load(file_path))
    net_copy=copy.deepcopy(net)
    net_copy2=copy.deepcopy(net)

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



    #--------------------------------------Instrumenting original learning----------------------------------------------------------
    mia_score=measure_mia(net, forget_loader, test_loader, device)
    retain_acc=test(net, retain_loader, device)
    forget_acc=test(net, forget_loader, device)
    train_acc=test(net, img_real_data_loader, device)
    test_acc=test(net, test_loader, device)
    net_r=copy.deepcopy(net)
    print('\nInstrumented Pretrained Model: ')
    print('======================================')
    print('mia_score: ', mia_score)
    print('retain_acc: ', retain_acc)
    print('forget_acc: ', forget_acc)
    print('train_acc: ', train_acc)
    print('test_acc: ', test_acc)
    print('======================================')

    stat_data = {
        'mia_score': [mia_score],
        'retain_acc': [retain_acc],
        'forget_acc': [forget_acc],
        'train_acc': [train_acc],
        'test_acc': [test_acc],
    }

    df = pd.DataFrame(stat_data)

    file_path = os.path.join(result_directory_path,'pretrained_instrumentation_MIAdefense.csv')
    df.to_csv(file_path, index=False)

    attack_result_pretrained=evaluate_mia(result_directory_path, net, img_real_data_loader, test_loader,case='pretrained')

    random_classes=class_sampler(train_loader=img_real_data_loader, samp_num_classes=6)


    condensive_inversion_multi(directory=result_directory_path, 
                    net=net,
                    ref_net=net_r,
                    training_images=train_images, 
                    training_labels=train_labels,
                    img_shape=(channel,im_size[0],im_size[1]),
                    num_classes=num_classes,
                    random_classes=random_classes,
                    samps_per_class=20,
                    hidden_size= 128,
                    lr_inverter=1e-3,
                    inverter_epochs= 200,
                    batch_size=64,
                    case= 'original',
                    device=device)

  
    #--------------------------Initializing my Unlearning Method--------------------------------------------------------------------------- 
    ref_net=copy.deepcopy(net)
    for param in list(ref_net.parameters()):
        param.requires_grad = False

    beggining=Beginning(ref_net).to(device)

    intermediate=Intermediate(ref_net).to(device)
 
    data_bank=Databank(beggining=beggining, intermediate=intermediate).to(device)
    file_path = os.path.join(new_directory_path,'databank.pth')
    data_bank.load_state_dict(torch.load(file_path))

    final=Final(ref_net).to(device)
    file_path = os.path.join(new_directory_path,'final.pth')
    final.load_state_dict(torch.load(file_path))

    combined_model=CombinedModel(databank=data_bank, final=final).to(device)


    optim_model=torch.optim.Adam(combined_model.parameters(), lr=lr_proposed)
    criterion = nn.CrossEntropyLoss()
    #----------------------------------------------------------------------------------------------------------------------------------



    #--------------------------my Unlearning Method------------------------------------------------------------------------------------
    
    starting_time = time.time()
    combined_model=modular_unlearning(combined_model, optim_model, criterion, device, beggining_epochs, intermediate_epochs, final_epochs, overture_epochs, final_thr, img_syn_loader, reduced_retain_loader)
    ending_time = time.time()
    unlearning_time=ending_time - starting_time
    mia_score=measure_mia(combined_model, forget_loader,test_loader, device)
    retain_acc=test(combined_model, retain_loader, device)
    forget_acc=test(combined_model, forget_loader, device)
    train_acc=test(combined_model, img_real_data_loader, device)
    test_acc=test(combined_model, test_loader, device)
    combined_model_copy=copy.deepcopy(combined_model)

    print('\nModular Unlearning Stats: ')
    print('======================================')
    print('mia_score: ', mia_score)
    print('retain_acc: ', retain_acc)
    print('forget_acc: ', forget_acc)
    print('train_acc: ', train_acc)
    print('test_acc: ', test_acc)
    print('======================================')

    stat_data = {
        'mia_score': [mia_score],
        'retain_acc': [retain_acc],
        'forget_acc': [forget_acc],
        'train_acc': [train_acc],
        'test_acc': [test_acc],
    }

    df = pd.DataFrame(stat_data)

    file_path = os.path.join(result_directory_path,'proposed_unlearning_MIAdefense.csv')
    df.to_csv(file_path, index=False)

    

    condensive_inversion_multi(directory=result_directory_path, 
                        net=combined_model,
                        ref_net=net_copy2,
                        training_images=train_images, 
                        training_labels=train_labels,
                        img_shape=(channel,im_size[0],im_size[1]),
                        num_classes=num_classes,
                        random_classes=random_classes,
                        samps_per_class=20,
                        hidden_size= 128,
                        lr_inverter=1e-3,
                        inverter_epochs= 200,
                        batch_size=64,
                        case= 'proposed',
                        device=device)
    
    
    attack_result_proposed=evaluate_mia(result_directory_path, combined_model_copy, img_real_data_loader, test_loader,case='proposed')


    #--------------------------------DP training---------------------------------------------------------------------------------------------
    vgg16=modify_vgg16(channel, im_size[0], num_classes)
    net=Vgg16(vgg16=vgg16).to(device)
    starting_time = time.time()
    dp_net=DP_Adam(net=net,train_loader=img_real_data_loader,max_grad_norm=MAX_GRAD_NORM, epsilon=EPSILON, delta=DELTA, multiplier=MULTIPLIER, train_lr=retrain_lr, train_epochs=retrain_epochs, device=device)
    ending_time = time.time()
    training_time=ending_time - starting_time

    mia_score=measure_mia(dp_net, forget_loader,test_loader, device)
    retain_acc=test(dp_net, retain_loader, device)
    forget_acc=test(dp_net, forget_loader, device)
    train_acc=test(dp_net, img_real_data_loader, device)
    test_acc=test(dp_net, test_loader, device)
    dp_net_copy=copy.deepcopy(dp_net)

    print('\nDP retraining Stats: ')
    print('======================================')
    print('mia_score: ', mia_score)
    print('retain_acc: ', retain_acc)
    print('forget_acc: ', forget_acc)
    print('train_acc: ', train_acc)
    print('test_acc: ', test_acc)
    print('======================================')

    stat_data = {
        'mia_score': [mia_score],
        'retain_acc': [retain_acc],
        'forget_acc': [forget_acc],
        'train_acc': [train_acc],
        'test_acc': [test_acc],
    }

    df = pd.DataFrame(stat_data)

    file_path = os.path.join(result_directory_path,'DP_retraining_MIAdefense.csv')
    df.to_csv(file_path, index=False)


    condensive_inversion_multi(directory=result_directory_path, 
                    net=dp_net,
                    ref_net=net_copy,
                    training_images=train_images, 
                    training_labels=train_labels,
                    img_shape=(channel,im_size[0],im_size[1]),
                    num_classes=num_classes,
                    random_classes=random_classes,
                    samps_per_class=20,
                    hidden_size= 128,
                    lr_inverter=1e-3,
                    inverter_epochs= 200,
                    batch_size=64,
                    case= 'DP',
                    device=device)
    
    attack_result_DP=evaluate_mia(result_directory_path, dp_net_copy, img_real_data_loader, test_loader,case='DP')

    # Assume attack_results_1 and attack_results_2 are your attack results
    attack_result_1 = attack_result_pretrained.get_result_with_max_auc()
    attack_result_2 = attack_result_proposed.get_result_with_max_auc()
    attack_result_3 = attack_result_DP.get_result_with_max_auc()

    # Get the ROC curve data
    roc_curve_data_1 = attack_result_1.roc_curve
    roc_curve_data_2 = attack_result_2.roc_curve
    roc_curve_data_3 = attack_result_3.roc_curve

    # Extract FPR, TPR for each attack result (you might need to adjust these lines based on actual implementation)
    fpr_1, tpr_1, _ = roc_curve_data_1.fpr, roc_curve_data_1.tpr, roc_curve_data_1.thresholds
    fpr_2, tpr_2, _ = roc_curve_data_2.fpr, roc_curve_data_2.tpr, roc_curve_data_2.thresholds
    fpr_3, tpr_3, _ = roc_curve_data_3.fpr, roc_curve_data_3.tpr, roc_curve_data_3.thresholds

    # Plotting the ROC curves
    plt.figure(figsize=(6, 6),dpi=150)
    plt.plot(fpr_1, tpr_1, label='Pretrained', color='blue')
    plt.plot(fpr_2, tpr_2, label='Proposed', color='Green')
    plt.plot(fpr_3, tpr_3, label='DP', color='Red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    file_path = os.path.join(result_directory_path,'MIA_roc_curve_comparison.png')
    # Save the figure
    plt.savefig(file_path, bbox_inches='tight')  # saves the figure to the current directory with the name 'roc_curve.png'

    # Show the figure
    plt.show()
    plt.show()





if __name__ == '__main__':
    main()