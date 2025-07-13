
import warnings

import torch.utils
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from utils.loading_model import get_model
from utils.utils import *
from sklearn.cluster import KMeans
from torchvision import datasets, transforms
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
from unlearn_methods.ssd import ssd_unlearn
from unlearn_methods.ssd_lf import ssdlf_unlearn
from unlearn_methods.accelerated_cf import Accelerated_CF_Unlearner
from utils.loading_model import get_model
from utils.lira_mia import LiRA_MIA

import argparse



def main(args):
    print("-----------------------------------------------")
    print(f"Experiment: {args.exp}")
    print(f"Model Name: {args.model_name}")
    print(f"Unlearning Mode: {args.unlearning_mode}")
    print("-----------------------------------------------")
    print("\n")

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

    #----------------------Hyperparameters---------------------------------
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")


    batch_size = args.batch_size
    channel = 3
    im_size = torch.load(os.path.join(data_storage,'im_size.pt'))
    num_classes = torch.load(os.path.join(data_storage,'num_classes.pt'))


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

    json_file_name=f'hyperparameters/{args.dataset}_{args.model_name}_hyperparameters.json'
    params = load_hyperparameters(json_file_name, args)
    globals().update(params)


    #------------------------------------------------------------------------



    #----------------------------Loading stuff------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()


    file_path = os.path.join(data_storage,'test_dataset.pth')
    dst_test = torch.load(file_path)

    file_path = os.path.join(data_storage,'train_dataset.pth')
    img_real_data_dataset = torch.load(file_path)

    img_real_data_loader=torch.utils.data.DataLoader(img_real_data_dataset, batch_size=batch_size, shuffle=True)


    test_loader=torch.utils.data.DataLoader(dst_test, batch_size=batch_size, shuffle=True)


    # divide dst_test into used and unused parts
    test_split_ratio = 0.7
    used_size=int(test_split_ratio*len(dst_test))
    dst_test_used, dst_test = torch.utils.data.random_split(dst_test, [used_size, len(dst_test)-used_size])
    test_loader=torch.utils.data.DataLoader(dst_test, batch_size=batch_size, shuffle=True)



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

 

    # _______________________________________________________________________________________________________________________________
    if do_acatf:
        #--------------------------Accelerated Catastrophic Forgetting Method - V1-----------------------------------------------------------
        naive_net=get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
        starting_time = time.time()
        #----------------------Unlearning with sampled dataset--------------------------------------------------------------------------------------
        unlearner = Accelerated_CF_Unlearner(
                        original_model=naive_net,
                        retain_dataloader=retain_loader,
                        forget_dataset=forget_loader.dataset,
                        test_dataset=dst_test_used,
                        weight_distribution=weight_distribution,
                        k=k,
                        K=K,
                        device=device
                        )


        model_unlearned = unlearner.train(epochs=af_epochs, 
                                        learning_rate=acf_lr)
        
        ending_time = time.time()
        unlearning_time=ending_time - starting_time
        mia_score=LiRA_MIA(model_unlearned, forget_loader, test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)
        retain_acc=test(model_unlearned, retain_loader, device)
        forget_acc=test(model_unlearned, forget_loader, device)
        test_acc=test(model_unlearned, test_loader, device)


        print('\nAccelerated CF - V1 Stats: ')
        print('======================================')
        print('mia_score: ', mia_score)
        print('retain_acc: ', retain_acc)
        print('forget_acc: ', forget_acc)
        print('test_acc: ', test_acc)
        print('unlearning_time: ', unlearning_time)
        print('======================================')


        stat_data = {
            'mia_score': [mia_score],
            'retain_acc': [retain_acc],
            'forget_acc': [forget_acc],
            'test_acc': [test_acc],
            'unlearning_time': [unlearning_time]
        }

        df = pd.DataFrame(stat_data)


        save_data(result_directory_path, 'acatf', df, args.model_name, args.exp, choice)
        #-------------------------------------------------------------------------------------------------------------------------------

    #_______________________________________________________________________________________________________________________________



    if do_retrain:
        #--------------------------Retraining Method------------------------------------------------------------------------------------
        naive_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=False)
        starting_time = time.time()
        naive_net= retraining(naive_net, 
                              criterion, 
                              device, 
                              retrain_lr, 
                              retrain_momentum,
                              retrain_weight_decay,
                              retrain_warmup,
                              retrain_epochs, 
                              retain_loader,
                              decreasing_lr="50,75"
                              )
        ending_time = time.time()
        unlearning_time=ending_time - starting_time

        mia_score=LiRA_MIA(naive_net, forget_loader,test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)
        retain_acc=test(naive_net, retain_loader, device)
        forget_acc=test(naive_net, forget_loader, device)
        test_acc=test(naive_net, test_loader, device)

        print('\nRetraining-Unlearning Stats: ')
        print('======================================')
        print('mia_score: ', mia_score)
        print('retain_acc: ', retain_acc)
        print('forget_acc: ', forget_acc)
        print('test_acc: ', test_acc)
        print('unlearning_time: ', unlearning_time)
        print('======================================')

        stat_data = {
            'mia_score': [mia_score],
            'retain_acc': [retain_acc],
            'forget_acc': [forget_acc],
            'test_acc': [test_acc],
            'unlearning_time': [unlearning_time]
        }

        df = pd.DataFrame(stat_data)

        save_data(result_directory_path, 'retraining', df, args.model_name, args.exp, choice)

    

    if do_CF:

        #----------------------Catastrophic Forgetting Method---------------------------------------------------------------------------------
        naive_net=get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
        starting_time = time.time()
        naive_net= retraining(naive_net, 
                              criterion, 
                              device, 
                              cf_lr, 
                              cf_momentum,
                              cf_weight_decay,
                              cf_warmup,
                              cf_epochs, 
                              retain_loader,
                              decreasing_lr="50,75"
                              )
        ending_time = time.time()
        unlearning_time=ending_time - starting_time

        mia_score=LiRA_MIA(naive_net, forget_loader,test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)
        retain_acc=test(naive_net, retain_loader, device)
        forget_acc=test(naive_net, forget_loader, device)
        test_acc=test(naive_net, test_loader, device)


        print('\nCatastrophic-Forgetting Stats: ')
        print('======================================')
        print('mia_score: ', mia_score)
        print('retain_acc: ', retain_acc)
        print('forget_acc: ', forget_acc)
        print('test_acc: ', test_acc)
        print('unlearning_time: ', unlearning_time)
        print('======================================')

        stat_data = {
            'mia_score': [mia_score],
            'retain_acc': [retain_acc],
            'forget_acc': [forget_acc],
            'test_acc': [test_acc],
            'unlearning_time': [unlearning_time]
        }

        df = pd.DataFrame(stat_data)

        save_data(result_directory_path, 'CF', df, args.model_name, args.exp, choice)



    if do_fisher:
        #----------------------Fischer-Forgetting Method---------------------------------------------------------------------------------
        naive_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
        starting_time = time.time()
        forgot_net=fisher_forgetting(naive_net, retain_loader, num_classes, device, class_to_forget=forgetfull_class if choice=='classwise' else None, num_to_forget=None, alpha=fisher_alpha)
        ending_time = time.time()
        unlearning_time=ending_time - starting_time

        mia_score=LiRA_MIA(forgot_net, forget_loader, test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)
        retain_acc=test(forgot_net, retain_loader, device)
        forget_acc=test(forgot_net, forget_loader, device)
        test_acc=test(forgot_net, test_loader, device)

        print('\nFisher-Forgetting Stats: ')
        print('======================================')
        print('mia_score: ', mia_score)
        print('retain_acc: ', retain_acc)
        print('forget_acc: ', forget_acc)
        print('test_acc: ', test_acc)
        print('unlearning_time: ', unlearning_time)
        print('======================================')

        stat_data = {
            'mia_score': [mia_score],
            'retain_acc': [retain_acc],
            'forget_acc': [forget_acc],
            'test_acc': [test_acc],
            'unlearning_time': [unlearning_time]
        }

        df = pd.DataFrame(stat_data)

        save_data(result_directory_path, 'fischerForgetting', df, args.model_name, args.exp, choice)

        del forgot_net
        del naive_net



    if do_ssd:
        #----------------------Synaptic Dampening Method---------------------------------------------------------------------------------
        total_dataset = torch.utils.data.ConcatDataset([forget_set_real, retain_set_real])
        total_loader = torch.utils.data.DataLoader(total_dataset, batch_size=batch_size, shuffle=True)
        naive_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
        starting_time = time.time()
        forgot_net=ssd_unlearn(naive_net, 
                               forget_loader,
                               total_loader, 
                               ssd_lr, 
                               exponent=exponent, 
                               lower_bound=lower_bound,
                               selective_weighting=selective_weighting, 
                               dampening_constant=dampening_constant,
                               device=device
                            )
        ending_time = time.time()
        unlearning_time=ending_time - starting_time

        mia_score=LiRA_MIA(forgot_net, forget_loader, test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)
        retain_acc=test(forgot_net, retain_loader, device)
        forget_acc=test(forgot_net, forget_loader, device)
        test_acc=test(forgot_net, test_loader, device)

        print('\nSynaptic-Dampening Stats: ')
        print('======================================')
        print('mia_score: ', mia_score)
        print('retain_acc: ', retain_acc)
        print('forget_acc: ', forget_acc)
        print('test_acc: ', test_acc)
        print('unlearning_time: ', unlearning_time)
        print('======================================')

        stat_data = {
            'mia_score': [mia_score],
            'retain_acc': [retain_acc],
            'forget_acc': [forget_acc],
            'test_acc': [test_acc],
            'unlearning_time': [unlearning_time]
        }

        df = pd.DataFrame(stat_data)

        save_data(result_directory_path, 'synaptic_dampening', df, args.model_name, args.exp, choice)

        del forgot_net
        del naive_net




    if do_ssd_lf:
        #----------------------Synaptic Dampening- Label Free Method---------------------------------------------------------------------------------
        total_dataset = torch.utils.data.ConcatDataset([forget_set_real, retain_set_real])
        total_loader = torch.utils.data.DataLoader(total_dataset, batch_size=batch_size, shuffle=True)
        naive_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
        starting_time = time.time()
        forgot_net = ssdlf_unlearn(naive_net, 
                                    forget_loader,
                                    total_loader, 
                                    ssd_lf_lr, 
                                    exponent=exponent_lf, 
                                    lower_bound=lower_bound_lf,
                                    selective_weighting=selective_weighting_lf, 
                                    dampening_constant=dampening_constant_lf,
                                    device=device
                                    )
        ending_time = time.time()
        unlearning_time=ending_time - starting_time

        mia_score=LiRA_MIA(forgot_net, forget_loader, test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)
        retain_acc=test(forgot_net, retain_loader, device)
        forget_acc=test(forgot_net, forget_loader, device)
        test_acc=test(forgot_net, test_loader, device)

        print('\nSynaptic-Dampening-Label Free Stats: ')
        print('======================================')
        print('mia_score: ', mia_score)
        print('retain_acc: ', retain_acc)
        print('forget_acc: ', forget_acc)
        print('test_acc: ', test_acc)
        print('unlearning_time: ', unlearning_time)
        print('======================================')

        stat_data = {
            'mia_score': [mia_score],
            'retain_acc': [retain_acc],
            'forget_acc': [forget_acc],
            'test_acc': [test_acc],
            'unlearning_time': [unlearning_time]
        }

        df = pd.DataFrame(stat_data)

        save_data(result_directory_path, 'synaptic_dampening_labelfree', df, args.model_name, args.exp, choice)





    if do_distillation:
        #-----------------------Distillation based Method---------------------------------------------------------------------------------
        teacher_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
        student_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
        starting_time = time.time()
        distilled_net = distillation_unlearning(retain_loader, 
                                                distill_lr ,
                                                distill_momentum,
                                                distill_weight_decay,
                                                student_net, 
                                                teacher_net, 
                                                distill_epochs, 
                                                device, 
                                                alpha=distill_hard_weight, 
                                                gamma=distill_soft_weight, 
                                                kd_T=distill_kdT,
                                                decreasing_lr="50,75"
                                                )
        ending_time = time.time()
        unlearning_time=ending_time - starting_time

        mia_score=LiRA_MIA(distilled_net, forget_loader, test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)
        retain_acc=test(distilled_net, retain_loader, device)
        forget_acc=test(distilled_net, forget_loader, device)
        test_acc=test(distilled_net, test_loader, device)

        print('\nDistillation-Unlearning Stats: ')
        print('======================================')
        print('mia_score: ', mia_score)
        print('retain_acc: ', retain_acc)
        print('forget_acc: ', forget_acc)
        print('test_acc: ', test_acc)
        print('unlearning_time: ', unlearning_time)
        print('======================================')

        stat_data = {
            'mia_score': [mia_score],
            'retain_acc': [retain_acc],
            'forget_acc': [forget_acc],
            'test_acc': [test_acc],
            'unlearning_time': [unlearning_time]
        }

        df = pd.DataFrame(stat_data)

        save_data(result_directory_path, 'distillation', df, args.model_name, args.exp, choice)



    if do_scrub:
        #-----------------------SCRUB Method---------------------------------------------------------------------------------
        student_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
        teacher_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
        starting_time = time.time()

        distilled_net = scrub_model(
                                    teacher_net,
                                    student_net,
                                    retain_loader,
                                    forget_loader,
                                    lr=scrub_lr,
                                    momentum=scrub_momentum,
                                    weight_decay=scrub_weight_decay,
                                    warmup= scrub_warmup,
                                    m_steps=scrub_msteps,
                                    epochs=scrub_epochs,
                                    kd_temp=scrub_kd_T,
                                    gamma=scrub_gamma,
                                    beta=scrub_beta,
                                    milestones=[5, 10, 15],
                                    device=device
                                )
        

        ending_time = time.time()
        unlearning_time=ending_time - starting_time

        mia_score=LiRA_MIA(distilled_net, forget_loader, test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)
        retain_acc=test(distilled_net, retain_loader, device)
        forget_acc=test(distilled_net, forget_loader, device)
        test_acc=test(distilled_net, test_loader, device)

        print('\nSCRUB-Unlearning Stats: ')
        print('======================================')
        print('mia_score: ', mia_score)
        print('retain_acc: ', retain_acc)
        print('forget_acc: ', forget_acc)
        print('test_acc: ', test_acc)
        print('unlearning_time: ', unlearning_time)
        print('======================================')

        stat_data = {
            'mia_score': [mia_score],
            'retain_acc': [retain_acc],
            'forget_acc': [forget_acc],
            'test_acc': [test_acc],
            'unlearning_time': [unlearning_time]
        }

        df = pd.DataFrame(stat_data)

        save_data(result_directory_path, 'SCRUB', df, args.model_name, args.exp, choice)


    if do_bad_distillation:
        #-----------------------Bad Teacher based Distillation Method---------------------------------------------------------------------------------
        teacher_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
        student_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
        bad_teacher_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=False)
        starting_time = time.time()
        # sample the retain_set_real by partial_retain_ratio (for bad distillation)
        partial_retain_set_real = torch.utils.data.Subset(retain_set_real, random.sample(range(len(retain_set_real)), int(len(retain_set_real)*partial_retain_ratio)))
        distilled_net=blindspot_unlearner(model=student_net, 
                                          unlearning_teacher=bad_teacher_net, 
                                          full_trained_teacher=teacher_net, 
                                          retain_data=partial_retain_set_real,
                                          forget_data=forget_set_real, 
                                          epochs = bad_distill_epochs,
                                          lr = bad_distill_lr,
                                          momentum = bad_momentum,
                                          weight_decay = bad_distill_weight_decay,
                                          batch_size = batch_size, 
                                          device = device, 
                                          KL_temperature = bad_kdT
                                          )
        
        ending_time = time.time()
        unlearning_time=ending_time - starting_time

        mia_score=LiRA_MIA(distilled_net, forget_loader, test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)
        retain_acc=test(distilled_net, retain_loader, device)
        forget_acc=test(distilled_net, forget_loader, device)
        test_acc=test(distilled_net, test_loader, device)

        print('\nBad Teacher Distillation-Unlearning Stats: ')
        print('======================================')
        print('mia_score: ', mia_score)
        print('retain_acc: ', retain_acc)
        print('forget_acc: ', forget_acc)
        print('test_acc: ', test_acc)
        print('unlearning_time: ', unlearning_time)
        print('======================================')

        stat_data = {
            'mia_score': [mia_score],
            'retain_acc': [retain_acc],
            'forget_acc': [forget_acc],
            'test_acc': [test_acc],
            'unlearning_time': [unlearning_time]
        }

        df = pd.DataFrame(stat_data)

        save_data(result_directory_path, 'Bad_distillation', df, args.model_name, args.exp, choice)


    if do_l1_sparsity:
        #-----------------------Sparsification based Method---------------------------------------------------------------------------------
        naive_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
        starting_time = time.time()
        sparsified_net = unlearn_with_l1_sparsity(
                                            naive_net, 
                                            retain_loader,
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
        retain_acc=test(sparsified_net, retain_loader, device)
        forget_acc=test(sparsified_net, forget_loader, device)
        test_acc=test(sparsified_net, test_loader, device)

        print('\nSparsification-Unlearning Stats: ')
        print('======================================')
        print('mia_score: ', mia_score)
        print('retain_acc: ', retain_acc)
        print('forget_acc: ', forget_acc)
        print('test_acc: ', test_acc)
        print('unlearning_time: ', unlearning_time)
        print('======================================')

        stat_data = {
            'mia_score': [mia_score],
            'retain_acc': [retain_acc],
            'forget_acc': [forget_acc],
            'test_acc': [test_acc],
            'unlearning_time': [unlearning_time]
        }

        df = pd.DataFrame(stat_data)

        save_data(result_directory_path, 'L1_sparsity', df, args.model_name, args.exp, choice)



    if do_pruning:
        #-----------------------Pruning based Method---------------------------------------------------
        naive_net= get_model(args.model_name, args.exp, data_storage, num_classes, device, load=True)
        starting_time = time.time()

        pruned_net = unlearn_with_pruning(
                                            naive_net,
                                            retain_loader, 
                                            unlearn_epochs=prune_epochs, 
                                            target_sparsity=prune_target_sparsity, 
                                            learning_rate=prune_lr,
                                            momentum = prune_momentum, 
                                            weight_decay=prune_weight_decay,
                                            prune_step=prune_step,
                                            decreasing_lr="50,75",
                                            print_freq=50,
                                            device=device)

        ending_time = time.time()
        unlearning_time=ending_time - starting_time

        mia_score=LiRA_MIA(pruned_net, forget_loader, test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)
        retain_acc=test(pruned_net, retain_loader, device)
        forget_acc=test(pruned_net, forget_loader, device)
        test_acc=test(pruned_net, test_loader, device)

        print('\nPrunning-Unlearning Stats: ')
        print('======================================')
        print('mia_score: ', mia_score)
        print('retain_acc: ', retain_acc)
        print('forget_acc: ', forget_acc)
        print('test_acc: ', test_acc)
        print('unlearning_time: ', unlearning_time)
        print('======================================')

        stat_data = {
            'mia_score': [mia_score],
            'retain_acc': [retain_acc],
            'forget_acc': [forget_acc],
            'test_acc': [test_acc],
            'unlearning_time': [unlearning_time]
        }

        df = pd.DataFrame(stat_data)

        save_data(result_directory_path, 'pruning', df, args.model_name, args.exp, choice)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=1, help="Experiment number (default: 1)")
    parser.add_argument('--model_name', type=str, choices=['cnn_s', 'resnet_s', 'resnetlarge_s'], required=True, 
                        help="Choose the model name from: vit, resnet, resnetlarge, vit_s, resnet_s, resnetlarge_s")
    parser.add_argument('--unlearning_mode', type=str, choices=['uniform', 'few_uniform', 'large_uniform', 'classwise', 'k_classwise'], required=True, 
                        help="Choose the unlearning mode from: uniform, or few_uniform, or classwise or k_classwise")
    parser.add_argument('--retrain_lr', type=float, default=1e-1, help="Retrain learning rate (default: 1e-4)")
    parser.add_argument('--dataset', type=str, choices=['cifar10','cifar100', 'svhn', 'cinic10'], default='cifar10', 
                        help="Choose the dataset from: cifar10 , cinic10, cifar100 or svhn")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (default: 128)")
    
    
    args = parser.parse_args()

    main(args)
