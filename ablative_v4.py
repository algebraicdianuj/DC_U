
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from utils.loading_model import get_model
from utils.dataset_sampler import *
# from dc_methods.dc_blend import blend_DC
from dc_methods.dc_blend_v2 import blend_DC
# from dc_methods.dc_blend_fast import blend_DC
# from dc_methods.dc_blend_fast_v2 import blend_DC
from utils.utils import *
import random
import argparse
import os


import warnings
warnings.filterwarnings('ignore')
import os
import torch
import torch.nn as nn
import numpy as np
import time
import numpy as np
import random
import time
import pandas as pd
import torch
import pickle


from utils.hyperparameters import load_hyperparameters
from unlearn_methods.retrain import retraining
from unlearn_methods.distillation import distillation_unlearning
from unlearn_methods.sparisification import unlearn_with_l1_sparsity
from unlearn_methods.prunning import unlearn_with_pruning
from unlearn_methods.bad_distillation import blindspot_unlearner
from unlearn_methods.ntk_scrubbing import ntk_scrubbing
from unlearn_methods.fisher_forget import fisher_forgetting
from unlearn_methods.scrub import scrub_model
from unlearn_methods.accelerated_cf import Accelerated_CF_Unlearner
from unlearn_methods.ssd import ssd_unlearn
from unlearn_methods.ssd_lf import ssdlf_unlearn
from utils.loading_model import get_model
from utils.lira_mia import LiRA_MIA
import argparse



def main(args):

    directory_name= 'reservoir'
    current_path = os.getcwd()  
    data_storage = os.path.join(current_path, directory_name)  
    
    if not os.path.exists(data_storage): 
        os.makedirs(data_storage) 
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


    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")


    batch_size = args.batch_size
    channel = 3
    im_size = torch.load(os.path.join(data_storage,'im_size.pt'))
    num_classes = torch.load(os.path.join(data_storage,'num_classes.pt'))
    criterion = nn.CrossEntropyLoss()


    # if args.dataset == 'cifar10':
    #     n_subclasses=1000

    # elif args.dataset == 'cifar100':
    #     n_subclasses=100

    # else:
    #     ValueError("Dataset should be cifar10 or cifar100")

    n_subclasses = args.ipc

    



    file_path = os.path.join(data_storage,'test_dataset.pth')
    dst_test = torch.load(file_path)

    file_path = os.path.join(data_storage,'train_dataset.pth')
    img_real_data_dataset = torch.load(file_path)


    file_path = os.path.join(data_storage,'means.pt')
    img_mean = torch.load(file_path)

    file_path = os.path.join(data_storage,'stds.pt')
    img_std = torch.load(file_path)


    train_loader=torch.utils.data.DataLoader(img_real_data_dataset, batch_size=batch_size, shuffle=True)

 
    test_loader=torch.utils.data.DataLoader(dst_test, batch_size=batch_size, shuffle=True)

    # divide dst_test into used and unused parts
    test_split_ratio = 0.7
    used_size=int(test_split_ratio*len(dst_test))
    dst_test_used, dst_test = torch.utils.data.random_split(dst_test, [used_size, len(dst_test)-used_size])
    test_loader=torch.utils.data.DataLoader(dst_test, batch_size=batch_size, shuffle=True)




    do_acatf=True
    do_retrain=True
    do_CF=True
    do_fisher=False
    do_ssd=True
    do_ssd_lf=True
    do_distillation=True
    do_scrub=True
    do_bad_distillation=True
    do_l1_sparsity=True
    do_pruning=True

    json_file_name=f'hyperparameters/{args.dataset}_{args.model_name}_cond_hyperparameters.json'
    params = load_hyperparameters(json_file_name, args)
    globals().update(params)





    #-----------------------------------Creating Subclasses---------------------------------------------------------
    
    
    
    train_images=img_real_data_dataset.images
    train_labels=img_real_data_dataset.labels
    
    num_samples = train_images.size(0)
    
    # Generate a random permutation of indices
    indices = torch.randperm(num_samples)

    # Shuffle training data using the shuffled indices
    train_images = train_images[indices]
    train_labels = train_labels[indices]

    indices=list(range(len(train_images)))
    
    feature_extractor = get_model(args.feature_model_name, args.exp, data_storage, num_classes, device, load=False)

    fine_lab_train, indices_train_wrt_finelabels,_ = create_sub_classes(train_images, 
                                                                    train_labels, 
                                                                    model=feature_extractor, 
                                                                    num_classes=num_classes, 
                                                                    sub_divisions=n_subclasses, 
                                                                    device=device)

    del feature_extractor


    #---------------------Forget Set and Retain Set----------------------------------------------------------------------
    dataset_size = len(img_real_data_dataset)
    split_ratio = 0.1   # forget-retain split ratio (in case of random forgetting)
    large_split_ratio = 0.5 # forget-retain split ratio (in case of random forgetting)
    few_split_ratio = 0.001 # forget-retain split ratio (in case of random forgetting)
    forgetfull_class = random.randint(0, num_classes-1)  # if choice=='classwise' then this is the class to forget
    forgetfull_class_list = [1,2,3]    # if choice=='k_classwise' then this is the list of classes to forget

    choice = args.unlearning_mode
    print("\n\n >>Splitting the dataset into forget and retain sets according to the choice: ", choice)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    if choice=='uniform':

        # Define split ratio and sizes
        split = int(split_ratio * dataset_size)

        # Split indices into two parts
        forget_indices = indices[:split]
        retain_indices = indices[split:]

        forget_images=train_images[forget_indices]
        forget_labels=train_labels[forget_indices]

        retain_images=train_images[retain_indices]
        retain_labels=train_labels[retain_indices]

    
    elif choice=='large_uniform':

        # Define split ratio and sizes
        split = int(large_split_ratio * dataset_size)

        # Split indices into two parts
        forget_indices = indices[:split]
        retain_indices = indices[split:]

        forget_images=train_images[forget_indices]
        forget_labels=train_labels[forget_indices]

        retain_images=train_images[retain_indices]
        retain_labels=train_labels[retain_indices]

    
    elif choice=='few_uniform':

        # Define split ratio and sizes
        split = int(few_split_ratio * dataset_size)

        # Split indices into two parts
        forget_indices = indices[:split]
        retain_indices = indices[split:]

        forget_images=train_images[forget_indices]
        forget_labels=train_labels[forget_indices]

        retain_images=train_images[retain_indices]
        retain_labels=train_labels[retain_indices]


    elif choice=='classwise':

        # Find indices of images that belong to class 'c'
        forget_indices = [i for i, label in enumerate(train_labels) if label == forgetfull_class]

        # Find indices of images that do not belong to class 'c'
        retain_indices = [i for i, label in enumerate(train_labels) if label != forgetfull_class]

        # Now, update the dataset and images/labels
        forget_images = train_images[forget_indices]
        forget_labels = train_labels[forget_indices]

        retain_images = train_images[retain_indices]
        retain_labels = train_labels[retain_indices]


    elif choice=='k_classwise':

        forget_indices = [i for i, label in enumerate(train_labels) if label in forgetfull_class_list]

        retain_indices = [i for i, label in enumerate(train_labels) if label not in forgetfull_class_list]


        # Now, update the dataset and images/labels
        forget_images = train_images[forget_indices]
        forget_labels = train_labels[forget_indices]

        retain_images = train_images[retain_indices]
        retain_labels = train_labels[retain_indices]



    else:
        raise ValueError('Invalid choice of choice')


    print('Ratio of Forget to Retain: ', len(forget_indices)/len(retain_indices))



    forget_set_real=TensorDatasett(forget_images, forget_labels)
    retain_set_real=TensorDatasett(retain_images, retain_labels)
    forget_loader=torch.utils.data.DataLoader(forget_set_real, batch_size=batch_size, shuffle=True)
    retain_loader=torch.utils.data.DataLoader(retain_set_real, batch_size=batch_size, shuffle=True)
    
    #------------------------------------------------------------------------


    # target images are the residual images of clusters that contain forget images
    free_images, free_labels, residual_images, residual_labels = seperated_dataset_sampling(indices_train_wrt_finelabels, train_images, train_labels, forget_indices)
    
    free_images , free_labels = torch.cat(free_images), torch.cat(free_labels)
 
    free_dataset = TensorDatasett(free_images, free_labels)
    free_loader = torch.utils.data.DataLoader(free_dataset, batch_size=batch_size, shuffle=True)
    
    residual_dataset = TensorDatasett(residual_images, residual_labels)
    residual_loader = torch.utils.data.DataLoader(residual_dataset, batch_size=batch_size, shuffle=True)
    
    r_images = torch.cat([free_images, residual_images])
    r_labels = torch.cat([free_labels, residual_labels])
    
    
    
        
    
    #=========================== Main Change=====================================================================

    condensed_dataset=TensorDatasett(free_images,free_labels)
    condensed_loader=torch.utils.data.DataLoader(condensed_dataset, batch_size=batch_size, shuffle=True)
    
    comp_ratio = len(condensed_loader.dataset)/len(retain_loader.dataset)
    print("\n\n>> Ratio of Reduced Retain Set to Retain Set: ", comp_ratio)
    print(f"Size of Condensed Set: {len(condensed_loader.dataset)}")
    print(f"Size of Retain Set: {len(retain_loader.dataset)}")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # ===========================================================================================================



    #-----------------------Sparsification based Method---------------------------------------------------------------------------------
    naive_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
    starting_time = time.time()
    sparsified_net = unlearn_with_l1_sparsity(
                                        naive_net, 
                                        condensed_loader,
                                        test_loader=None,
                                        unlearn_epochs=l1_epochs, 
                                        learning_rate=l1_lr,
                                        momentum=l1_momentum, 
                                        weight_decay=l1_weight_decay,
                                        alpha=l1_alpha,
                                        no_l1_epochs=l1_no_l1_epochs,
                                        warmup=l1_warmup,
                                        decreasing_lr="50,75",
                                        print_freq=50,
                                        device=device)
    

    ending_time = time.time()
    unlearning_time=ending_time - starting_time

    mia_score=LiRA_MIA(sparsified_net, forget_loader, test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)
    free_retain_acc=test(sparsified_net, free_loader, device)
    residual_retain_acc=test(sparsified_net, residual_loader, device)
    forget_acc=test(sparsified_net, forget_loader, device)
    test_acc=test(sparsified_net, test_loader, device)

    print('\nSparsification-Unlearning Stats (Condensed): ')
    print('======================================')
    print('mia_score: ', mia_score)
    print('free retain acc: ', free_retain_acc)
    print('residual retain acc: ', residual_retain_acc)
    print('forget_acc: ', forget_acc)
    print('test_acc: ', test_acc)
    print('unlearning_time: ', unlearning_time)
    print('======================================')

    stat_data = {
        'mia_score': [mia_score],
        'free_retain_acc': [free_retain_acc],
        'residual_retain_acc': [residual_retain_acc],
        'forget_acc': [forget_acc],
        'test_acc': [test_acc],
        'unlearning_time': [unlearning_time]
    }

    df = pd.DataFrame(stat_data)

    save_data(result_directory_path, 'L1_sparsity_ablv4', df, args.model_name, args.exp, choice)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=1, help="Experiment number (default: 1)")
    parser.add_argument('--model_name', type=str, choices=['cnn_s', 'vit_s', 'resnet_s', 'resnetlarge_s'], required=True, 
                        help="Choose the model name from: vit, resnet, resnetlarge, vit_s, resnet_s, resnetlarge_s")
    parser.add_argument('--feature_model_name', type=str, choices=['cnn_s', 'resnet_s', 'vit_s', 'resnetlarge_s'], default='cnn_s',
                    help="Choose the model name")
    parser.add_argument('--unlearning_mode', type=str, choices=['uniform', 'few_uniform', 'large_uniform', 'classwise', 'k_classwise'], required=True, 
                        help="Choose the unlearning mode from: uniform, or few_uniform, or classwise or k_classwise")
    parser.add_argument('--retrain_lr', type=float, default=1e-1, help="Retrain learning rate (default: 1e-4)")
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'svhn', 'cinic10'], default='cifar10', 
                        help="Choose the dataset from: cifar10 , cinic10, cifar100 or svhn")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (default: 128)")
    parser.add_argument('--ipc', type=int, default=1000, help="Number of condensed images per class")
    
    
    args = parser.parse_args()

    main(args)
