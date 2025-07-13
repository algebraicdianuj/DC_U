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

import argparse
from utils.utils import *
from utils.loading_model import get_model
from utils.lira_mia import LiRA_MIA
from utils.dataset_sampler import *
from dc_methods.gradient_matching import GM_condensation
from dc_methods.distribution_matching import DM_condensation
from dc_methods.i_distribution_matching import IDM_condensation
# from dc_methods.dc_blend import blend_DC
from dc_methods.dc_blend_v2 import blend_DC
# from dc_methods.dc_blend_fast import blend_DC
# from dc_methods.dc_blend_fast_v2 import blend_DC
from train import train_model
from rich import print
from utils.dc_utils import (
    get_loops, get_dataset, get_network, get_time,
    DiffAugment, ParamDiffAug
)
import copy
from torchvision.utils import save_image



def main(args):
    print("-----------------------------------------------")
    print(f"Experiment: {args.exp}")
    print(f"Model Name: {args.model_name}")
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


    batch_size = 256
    channel = 3
    im_size = torch.load(os.path.join(data_storage,'im_size.pt'))
    num_classes = torch.load(os.path.join(data_storage,'num_classes.pt'))
    n_subclasses= args.ipc   # number of subclasses per class

    cifar_label_map = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }


    #----------------------------Loading stuff------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()


    file_path = os.path.join(data_storage,'test_dataset.pth')
    dst_test = torch.load(file_path)

    file_path = os.path.join(data_storage,'train_dataset.pth')
    img_real_data_dataset = torch.load(file_path)


    file_path = os.path.join(data_storage,'means.pt')
    img_mean = torch.load(file_path)

    file_path = os.path.join(data_storage,'stds.pt')
    img_std = torch.load(file_path)


    img_real_data_loader=torch.utils.data.DataLoader(img_real_data_dataset, batch_size=batch_size, shuffle=True)

    test_loader=torch.utils.data.DataLoader(dst_test, batch_size=batch_size, shuffle=True)


    do_blend_DC = True
    do_gm_DC = True
    do_dm_DC = True
    do_idm_DC = True

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


    #_______________________________________________________________________________________________________________________________
    #-------------------------Dataset Condensation based Forgetting Method-----------------------------------------------------------

    # target images are the residual images of clusters that contain forget images
    target_images, target_labels = dataset_sampling(indices_train_wrt_finelabels,train_images, train_labels)


    # save the target images and labels
    torch.save(target_images, os.path.join(data_storage, 'target_images.pt'))
    torch.save(target_labels, os.path.join(data_storage, 'target_labels.pt'))


    if do_blend_DC:
        lr_b = -2.397e-06*(args.ipc)**3 + 0.01265*(args.ipc)**2 - 0.2558*(args.ipc) + 1.461
        num_iterations_b = 20
        #-------------------------------------------------------------------------------------------------------------------------------
        
        net=get_model(args.feature_model_name, args.exp, data_storage, num_classes, device, load=False)


        starting_time = time.time()
        # ----------------------Blend Dataset Condensation---------------------------------------------------------------------------------
        condensed_images, condensed_labels = blend_DC(
                                                    target_images,
                                                    target_labels,
                                                    net,
                                                    lr=lr_b,
                                                    num_iterations=num_iterations_b, 
                                                    device=device,
                                                    batch_size=32
                                                    )
        ending_time = time.time()
        
        if args.save_img:
            save_cond_imgs(condensed_images, condensed_labels, n_subclasses, channel, img_mean, img_std, cifar_label_map, 'blend', args.exp, args.model_name)
        #-------------------------------------------------------------------------------------------------------------------------------------------

        condensed_dataset=TensorDatasett(condensed_images, condensed_labels)
        condensed_loader=torch.utils.data.DataLoader(condensed_dataset, batch_size=batch_size, shuffle=True)


        # #----------------------Unlearning with condensed dataset--------------------------------------------------------------------------------------
        net=get_model(args.model_name, args.exp, data_storage, num_classes, device, load=False)
        trained_net = train_model(net, condensed_loader, test_loader, criterion, device,
                                    lr=0.01, momentum=0.9, weight_decay=5e-4,
                                    epochs=40, warmup=0, save_dir=data_storage,
                                    model_name=args.model_name, exp=args.exp)
        
        
        cond_time=ending_time - starting_time
        train_acc = test(trained_net, img_real_data_loader, device)
        test_acc=test(trained_net, test_loader, device)
        mia_score=LiRA_MIA(trained_net,img_real_data_loader, test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)

        print('\nBlend Condensation CF Stats: ')
        print('======================================')
        print('mia_score: ', mia_score)
        print('train_acc: ', train_acc)
        print('test_acc: ', test_acc)
        print('condensation_time: ', cond_time)
        print('======================================')


        stat_data = {
            'mia_score': [mia_score],
            'train_acc': [train_acc],
            'test_acc': [test_acc],
            'condensation_time': [cond_time]
        }

        df = pd.DataFrame(stat_data)



        save_data2(result_directory_path, 'blend_condensation_CF', df, args.model_name, args.exp, args.ipc)



    

    if do_gm_DC:
        #-------------------------------------------------------------------------------------------------------------------------------
        images_all = torch.cat(target_images, dim=0)
        labels_all = torch.cat(target_labels, dim=0)
        model_fn= lambda: get_model(args.model_name, args.exp, data_storage, num_classes, device, load=False)

        starting_time = time.time()
        # ----------------------Blend Dataset Condensation---------------------------------------------------------------------------------
        condensed_images, condensed_labels = GM_condensation(model_fn, 
                                                            images_all, 
                                                            labels_all, 
                                                            criterion,
                                                            num_iterations=1000,
                                                            ipc=n_subclasses,
                                                            lr_img=0.1,
                                                            lr_net=0.01,
                                                            channel=channel, 
                                                            im_size=im_size, 
                                                            num_classes=num_classes,
                                                            dis_metric='ours',
                                                            bn=False,
                                                            do_dsa=True,
                                                            dsa_strategy='color_crop_cutout_flip_scale_rotate',
                                                            dsa_param= ParamDiffAug(),
                                                            batch_real=256,
                                                            batch_train=256,
                                                            mean=img_mean, 
                                                            std=img_std,
                                                            device=device
                                                            )
        ending_time = time.time()
        
        if args.save_img:
            save_cond_imgs(condensed_images, condensed_labels, n_subclasses, channel, img_mean, img_std, cifar_label_map,'gm', args.exp, args.model_name)
        #-------------------------------------------------------------------------------------------------------------------------------------------

        condensed_dataset=TensorDatasett(condensed_images, condensed_labels)
        condensed_loader=torch.utils.data.DataLoader(condensed_dataset, batch_size=batch_size, shuffle=True)


        # #----------------------Unlearning with condensed dataset--------------------------------------------------------------------------------------
        net=get_model(args.model_name, args.exp, data_storage, num_classes, device, load=False)
        trained_net = train_model(net, condensed_loader, test_loader, criterion, device,
                                    lr=0.01, momentum=0.9, weight_decay=5e-4,
                                    epochs=10, warmup=0, save_dir=data_storage,
                                    model_name=args.model_name, exp=args.exp)
        
        
        cond_time=ending_time - starting_time
        train_acc = test(trained_net, img_real_data_loader, device)
        test_acc=test(trained_net, test_loader, device)
        mia_score=LiRA_MIA(trained_net,img_real_data_loader,test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)

        print('\nGradeint Matching Condensation CF Stats: ')
        print('======================================')
        print('mia_score: ', mia_score)
        print('train_acc: ', train_acc)
        print('test_acc: ', test_acc)
        print('condensation_time: ', cond_time)
        print('======================================')


        stat_data = {
            'mia_score': [mia_score],
            'train_acc': [train_acc],
            'test_acc': [test_acc],
            'condensation_time': [cond_time]
        }

        df = pd.DataFrame(stat_data)



        save_data2(result_directory_path, 'gm_condensation_CF', df, args.model_name, args.exp, args.ipc)




    if do_dm_DC:
        #-------------------------------------------------------------------------------------------------------------------------------
        images_all = torch.cat(target_images, dim=0)
        labels_all = torch.cat(target_labels, dim=0)
        model_fn= lambda: get_model(args.model_name, args.exp, data_storage, num_classes, device, load=False)

        starting_time = time.time()
        # ----------------------Blend Dataset Condensation---------------------------------------------------------------------------------

        condensed_images, condensed_labels = DM_condensation(model_fn, 
                                                        images_all, 
                                                        labels_all, 
                                                        num_iterations=20000,
                                                        ipc=n_subclasses,
                                                        lr_img=1e-2,
                                                        channel=channel, 
                                                        im_size=im_size, 
                                                        num_classes=num_classes,
                                                        bn=False,
                                                        do_dsa=True,
                                                        dsa_strategy='color_crop_cutout_flip_scale_rotate',
                                                        dsa_param = ParamDiffAug(),
                                                        batch_real=256,
                                                        mean=img_mean, 
                                                        std=img_std,
                                                        device=device
                                                        )
        ending_time = time.time()
        
        if args.save_img:
            save_cond_imgs(condensed_images, condensed_labels, n_subclasses, channel, img_mean, img_std, cifar_label_map,'dm', args.exp, args.model_name)
        #-------------------------------------------------------------------------------------------------------------------------------------------

        condensed_dataset=TensorDatasett(condensed_images, condensed_labels)
        condensed_loader=torch.utils.data.DataLoader(condensed_dataset, batch_size=batch_size, shuffle=True)


        # #----------------------Unlearning with condensed dataset--------------------------------------------------------------------------------------
        net=get_model(args.model_name, args.exp, data_storage, num_classes, device, load=False)
        trained_net = train_model(net, condensed_loader, test_loader, criterion, device,
                                    lr=0.01, momentum=0.9, weight_decay=5e-4,
                                    epochs=10, warmup=0, save_dir=data_storage,
                                    model_name=args.model_name, exp=args.exp)
        
        
        cond_time=ending_time - starting_time
        train_acc = test(trained_net, img_real_data_loader, device)
        test_acc=test(trained_net, test_loader, device)
        mia_score=LiRA_MIA(trained_net,img_real_data_loader, test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)

        print('\nDistribution Matching Condensation CF Stats: ')
        print('======================================')
        print('mia_score: ', mia_score)
        print('train_acc: ', train_acc)
        print('test_acc: ', test_acc)
        print('condensation_time: ', cond_time)
        print('======================================')


        stat_data = {
            'mia_score': [mia_score],
            'train_acc': [train_acc],
            'test_acc': [test_acc],
            'condensation_time': [cond_time]
        }

        df = pd.DataFrame(stat_data)



        save_data2(result_directory_path, 'dm_condensation_CF', df, args.model_name, args.exp, args.ipc)


    
    if do_idm_DC:
    
        #-------------------------------------------------------------------------------------------------------------------------------
        images_all = torch.cat(target_images, dim=0)
        labels_all = torch.cat(target_labels, dim=0)
        model_fns= [lambda: get_model(args.model_name, ex, data_storage, num_classes, device, load=False) for ex in range(1, 4)]


        starting_time = time.time()
        # ----------------------Blend Dataset Condensation---------------------------------------------------------------------------------
        condensed_images, condensed_labels = IDM_condensation(
                                                            model_fns,
                                                            images_all,
                                                            labels_all,
                                                            ipc=n_subclasses,
                                                            channel=channel,
                                                            im_size=im_size,
                                                            num_classes=num_classes,
                                                            lr_img=1e-2,
                                                            num_iterations=20000,
                                                            do_dsa=True,
                                                            do_aug=False,
                                                            dsa_strategy='color_crop_cutout_flip_scale_rotate',
                                                            dsa_param= ParamDiffAug(),
                                                            batch_real=256,
                                                            aug_num=1,
                                                            ce_weight=0.1,
                                                            syn_ce=False,
                                                            mean=img_mean,
                                                            std=img_std,
                                                            device=device
                                                        )
        ending_time = time.time()
        
        if args.save_img:
            save_cond_imgs(condensed_images, condensed_labels, n_subclasses, channel, img_mean, img_std, cifar_label_map, 'idm', args.exp, args.model_name)

        #-------------------------------------------------------------------------------------------------------------------------------------------

        condensed_dataset=TensorDatasett(condensed_images, condensed_labels)
        condensed_loader=torch.utils.data.DataLoader(condensed_dataset, batch_size=batch_size, shuffle=True)


        # #----------------------Unlearning with condensed dataset--------------------------------------------------------------------------------------
        net=get_model(args.model_name, args.exp, data_storage, num_classes, device, load=False)
        trained_net = train_model(net, condensed_loader, test_loader, criterion, device,
                                    lr=0.01, momentum=0.9, weight_decay=5e-4,
                                    epochs=10, warmup=0, save_dir=data_storage,
                                    model_name=args.model_name, exp=args.exp)
        
        
        cond_time=ending_time - starting_time
        train_acc = test(trained_net, img_real_data_loader, device)
        test_acc=test(trained_net, test_loader, device)
        mia_score=LiRA_MIA(trained_net,img_real_data_loader,test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)

        print('\nIDM Condensation CF Stats: ')
        print('======================================')
        print('mia_score: ', mia_score)
        print('train_acc: ', train_acc)
        print('test_acc: ', test_acc)
        print('condensation_time: ', cond_time)
        print('======================================')


        stat_data = {
            'mia_score': [mia_score],
            'train_acc': [train_acc],
            'test_acc': [test_acc],
            'condensation_time': [cond_time]
        }

        df = pd.DataFrame(stat_data)



        save_data2(result_directory_path, 'idm_condensation_CF', df, args.model_name, args.exp, args.ipc)
    
    
    
    




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=1, help="Experiment number (default: 1)")
    parser.add_argument('--model_name', type=str, choices=['cnn_s', 'resnet_s', 'resnetlarge_s'], required=True, 
                        help="Choose the model name")
    parser.add_argument('--feature_model_name', type=str, choices=['cnn_s', 'resnet_s', 'resnetlarge_s'], default='cnn_s',
                        help="Choose the model name")
    parser.add_argument('--save_img', action='store_true', help="Save condensed images")
    parser.add_argument('--ipc', type=int, default=1, help="Number of subclasses per class (default: 1)")
    args = parser.parse_args()
    main(args)

