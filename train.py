
import warnings
warnings.filterwarnings('ignore')
import os
import torch
import torch.nn as nn
import numpy as np
import time
import pandas as pd
from utils.utils import *
from utils.loading_model import get_model
import argparse
import tempfile
import shutil
from utils.lira_mia import LiRA_MIA

def train_model(model, train_loader, val_loader, criterion, device, 
                lr=0.1, momentum=0.9, weight_decay=5e-4, 
                epochs=200, warmup=10, save_dir='./checkpoints', 
                model_name='model', exp='1'):
    

    temp_dir = tempfile.mkdtemp()
    temp_best_path = os.path.join(temp_dir, "best_model.pth")
    
    best_acc = 0
    all_result = {"train_acc": [], "val_acc": []}
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    
    # Use lambda scheduler with warmup and cosine annealing as in reference
    lambda0 = lambda cur_iter: (cur_iter + 1) / warmup if cur_iter < warmup else (
        0.5 * (1.0 + np.cos(np.pi * ((cur_iter - warmup) / (epochs - warmup))))
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    
    starting_time = time.time()
    
    for epoch in range(epochs):
        start_time = time.time()
        print(f"Current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
        
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0 or batch_idx + 1 == len(train_loader):
                print(f'Epoch: [{epoch+1}/{epochs}][{batch_idx+1}/{len(train_loader)}] '
                      f'Loss: {train_loss/(batch_idx+1):.4f}')
        
        # Calculate accuracies using the test function
        train_acc = test(model, train_loader, device)
        val_acc = test(model, val_loader, device)
        
        all_result["train_acc"].append(train_acc)
        all_result["val_acc"].append(val_acc)
        
        # Remember best accuracy and save best weights to temporary directory
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            torch.save(model.state_dict(), temp_best_path)
            print(f'New best validation accuracy: {best_acc:.3f}')
        
        # # Update learning rate
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, Time: {time.time() - start_time:.2f}s')
    
    if os.path.exists(temp_best_path):
        model.load_state_dict(torch.load(temp_best_path))
    
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'pretrained_{model_name}_exp_{exp}.pth')
    torch.save(model.state_dict(), file_path)
    
    training_time = time.time() - starting_time
    print(f'Total training time: {training_time:.2f}s, Best validation accuracy: {best_acc:.3f}')

    shutil.rmtree(temp_dir)
    
    return model                




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

    #----------------------Hyperparameters---------------------------------
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_epochs = args.epochs
    batch_size = args.batch_size
    channel = 3
    im_size = torch.load(os.path.join(data_storage,'im_size.pt'))
    num_classes = torch.load(os.path.join(data_storage,'num_classes.pt'))
    lr_net=args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    warmup = args.warmup
    #----------------------Hyperparameters---------------------------------


    file_path = os.path.join(data_storage,'train_dataset.pth')
    dst_train=torch.load(file_path)
    file_path = os.path.join(data_storage,'test_dataset.pth')
    dst_test=torch.load(file_path)
    train_loader=torch.utils.data.DataLoader(dst_train, batch_size=batch_size, shuffle=True)
    test_loader=torch.utils.data.DataLoader(dst_test, batch_size=batch_size, shuffle=True)


    #------------------Train the Net--------------------------------------------------------------------------------------------------------
    net=get_model(args.model_name, args.exp, data_storage, num_classes, device, load=False)
    criterion = nn.CrossEntropyLoss()

    starting_time = time.time()

    net = train_model(net, train_loader, test_loader, criterion, device,
                        lr=lr_net, momentum=momentum, weight_decay=weight_decay,
                        epochs=training_epochs, warmup=warmup, save_dir=data_storage,
                        model_name=args.model_name, exp=args.exp)
    
    ending_time = time.time()
    training_time=ending_time - starting_time

    file_path = os.path.join(data_storage, f'pretrained_{args.model_name}_exp_{args.exp}.pth')
    torch.save(net.state_dict(), file_path)

    train_acc=test(net, train_loader, device)
    test_acc=test(net, test_loader, device)

    mia_score=LiRA_MIA(net, train_loader, test_loader, nn.CrossEntropyLoss(reduction='none'), num_classes, device)


    stat_data = {
        'Training Time': [training_time],
        'Train Accuracy': [train_acc],
        'Test Accuracy': [test_acc],
        'MIA Score': [mia_score]
    }

    df = pd.DataFrame(stat_data)
    
    file_path = os.path.join(result_directory_path, f'stat_{args.model_name}_'+str(training_epochs)+f'epochs_exp_{args.exp}.csv')
    df.to_csv(file_path, index=False)




if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--exp',type=int, default=1, help='Experiment number')
    parser.add_argument('--model_name', type=str, choices=['cnn_s', 'vit_s', 'resnet_s', 'resnetlarge_s'], required=True, 
                        help="Choose the model name from: vit, resnet, resnetlarge, vit_s, resnet_s, resnetlarge_s")
    parser.add_argument('--batch_size', type=int, default=64, help="Choose the batch size")
    parser.add_argument('--epochs', type=int, default=200, help="Choose the number of epochs")
    parser.add_argument('--lr', type=float, default=1e-1, help="Choose the learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="Choose the momentum")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="Choose the weight decay")
    parser.add_argument('--warmup', type=int, default=10, help="Choose the warmup period")
    args=parser.parse_args()
    main(args)

