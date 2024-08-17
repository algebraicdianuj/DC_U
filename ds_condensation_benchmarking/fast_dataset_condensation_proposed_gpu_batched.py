
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
import pandas as pd
import os

def main():
    # Define your network architecture (MLP)
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size*4)
            self.fc2 = nn.Linear(hidden_size*4, hidden_size*2)
            self.fc3 = nn.Linear(hidden_size*2, hidden_size)
            self.fc4 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten input
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            y = F.relu(self.fc3(x))
            z = self.fc4(y)
            return z
        
        def feature(self, x):
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            y = F.relu(self.fc3(x))
            return y




    def test(model, data_loader, device):
        model.to(device)
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    def get_time():
        return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))

    batch_size = 256
    num_classes = 10
    batch_real = 256
    channel = 3
    im_size = (32, 32)
    hidden_size=128
    batcher=10


    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dst_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dst_train, batch_size=batch_size,
                                            shuffle=True, num_workers=2)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    im_size = (32, 32)
    channel = 3

    print('Running GPU batched Version of FDM-version 1...\n\n')

    #------------------Train the Net--------------------------------
    net= MLP(input_size=channel * im_size[0] * im_size[1], hidden_size=hidden_size, output_size=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    if not os.path.exists('model.pth'):
        optimizer=torch.optim.Adam(net.parameters(), lr=1e-3)
        # Teacher training before starting the dataset distillation process
        for epochy in range(30):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                output = net(data) 
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        del loss
        del output

        # save the trained model
        torch.save(net.state_dict(), 'model.pth')
        net.eval()

    else:
        net.load_state_dict(torch.load('model.pth'))
        net.eval()


    


    class BatchWeightedAverage(nn.Module):
        def __init__(self, batch_size, num_images_per_batch, channels, height, width):
            super(BatchWeightedAverage, self).__init__()
            self.weights = nn.Parameter(1/num_images_per_batch * torch.ones(batch_size, num_images_per_batch, device=device))
            self.channels = channels
            self.height = height
            self.width = width

        def forward(self, imgs):
            # imgs shape: (batch_size, num_images_per_batch, channels, height, width)
            imgs = imgs.view(imgs.shape[0], imgs.shape[1], -1)
            weighted_imgs = imgs * self.weights.unsqueeze(-1)
            weighted_imgs = torch.sum(weighted_imgs, dim=1)
            weighted_imgs = weighted_imgs.reshape(imgs.shape[0], self.channels, self.height, self.width)
            return weighted_imgs
        



    def Average(ref_imgs_all_batched, pretrained=net, channels = 3, height = 32, width = 32, num_epochs=100):

        ref_imgs_all_batched=ref_imgs_all_batched.to(device)
        weighted_avg_module = BatchWeightedAverage(batch_size=ref_imgs_all_batched.shape[0],
                                                    num_images_per_batch=ref_imgs_all_batched.shape[1],
                                                     channels=channels,
                                                    height=height, 
                                                    width=width).to(device)
        
        optim_weighted_avg = torch.optim.Adam(weighted_avg_module.parameters(), lr=1e-3)
        ref_features= pretrained.feature(ref_imgs_all_batched.view(-1, ref_imgs_all_batched.shape[2], ref_imgs_all_batched.shape[3], ref_imgs_all_batched.shape[4])).detach()

        for ep in range(num_epochs):
            fused_img_batch= weighted_avg_module(ref_imgs_all_batched)
            fused_img_features_batch= pretrained.feature(fused_img_batch)
            loss=torch.sum((torch.mean(ref_features, dim=0) - torch.mean(fused_img_features_batch, dim=0))**2)
            optim_weighted_avg.zero_grad()
            loss.backward()
            optim_weighted_avg.step()

        averaged_img=weighted_avg_module(ref_imgs_all_batched).detach()

        return averaged_img
        



    def extract_features(model, dataloader, device):
        features = []
        labels = []
        
        model.eval()
        with torch.no_grad():
            for data, label in dataloader:
                data = data.to(device)
                # feature = model(data)
                feature = model.feature(data)
                feature = feature.view(feature.size(0), -1)  # Flatten spatial dimensions
                features.append(feature.cpu())
                labels.append(label)
        
        return torch.cat(features, 0), torch.cat(labels, 0)



    def kmeans_pytorch(X, num_clusters, num_iterations=100, tol=1e-4):
        N, D = X.shape
        
        # Randomly initialize cluster centers
        C = X[torch.randperm(N)[:num_clusters]]
        
        for i in range(num_iterations):
            # Compute distances
            distances = torch.cdist(X, C)
            
            # Assign points to nearest cluster
            labels = torch.argmin(distances, dim=1)
            
            # Update cluster centers
            new_C = torch.stack([X[labels == k].mean(dim=0) for k in range(num_clusters)])
            
            # Check for convergence
            if torch.abs(new_C - C).sum() < tol:
                break
            
            C = new_C
        
        
        return labels


    def create_sub_classes(tensor, labels, model, num_classes=10, sub_divisions=10):
        
        new_labels = torch.zeros_like(labels)
        original_labels_dict = {}
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        dataset = TensorDataset(tensor, labels)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        features, _ = extract_features(model, loader, device)
        
        for i in range(num_classes):
            mask = labels == i
            class_features = features[mask]
            
            # Apply k-means clustering using PyTorch
            class_new_labels = kmeans_pytorch(class_features, num_clusters=sub_divisions)
            
            # Assign new labels
            new_subclass_labels = i * sub_divisions + class_new_labels
            new_labels[mask] = new_subclass_labels

            # Store original label reference
            for j in range(sub_divisions):
                original_labels_dict[int(i * sub_divisions + j)] = i
        
   
        return new_labels, original_labels_dict
    


    def correct_sizes(collected_ref_imgs_all,chann, height, width):
        batch_sizes=[]
        for i in range(len(collected_ref_imgs_all)):
            batch_sizes.append(collected_ref_imgs_all[i].shape[0])

        max_batch_size=max(batch_sizes)
        for i in range(len(collected_ref_imgs_all)):
            if collected_ref_imgs_all[i].shape[0]<max_batch_size:
                diff=max_batch_size-collected_ref_imgs_all[i].shape[0]
                collected_ref_imgs_all[i]=torch.cat([collected_ref_imgs_all[i], torch.zeros(diff, chann, height, width)], dim=0)


        return collected_ref_imgs_all
        


    training_images=[]
    training_labels=[]

    for batch in train_loader:
        training_images.append(batch[0])
        training_labels.append(batch[1])



    training_images=torch.cat(training_images, dim=0)
    training_labels=torch.cat(training_labels, dim=0)



    number_subclasses=[1,10,50]
    running_time=[]
    test_accs=[]
    ipc_record=[]

    for exp in range(len(number_subclasses)):
        print('\n================== Exp %d : IPC: %d =================='%(exp, number_subclasses[exp]))
        starting_time = time.time()
        n_subclasses=number_subclasses[exp]
        new_lab_train, original_labels_dict = create_sub_classes(training_images, training_labels, model=net, num_classes=num_classes, sub_divisions=n_subclasses)

        inverted_IMG=[]
        inverted_LABEL=[]
        indices_train_wrt_bucket=[]
        bucket_labbies=torch.unique(new_lab_train).tolist()
        collected_ref_imgs_all=[]


        for idx in bucket_labbies:

            indices_idx = torch.where(new_lab_train.to(device)==idx)[0]

            indices_train_wrt_bucket.append(indices_idx.cpu())


            ref_imgs_all = training_images[indices_idx.cpu()]
            collected_ref_imgs_all.append(ref_imgs_all)


            if len(collected_ref_imgs_all)==batcher or idx==bucket_labbies[-1]:
                collected_ref_imgs_all=correct_sizes(collected_ref_imgs_all, channel, im_size[0], im_size[1])
                ref_imgs_all_batched=torch.stack(collected_ref_imgs_all)

                inverted_image_batch = Average(ref_imgs_all_batched, pretrained=net, num_epochs=100)
                inverted_IMG.append(inverted_image_batch)
                collected_ref_imgs_all=[]



            ref_labs_all = training_labels[indices_idx.cpu()]
            inverted_LABEL.append(ref_labs_all[0])
        

        inverted_IMG=torch.cat(inverted_IMG, dim=0).cpu()
        inverted_LABEL=torch.tensor(inverted_LABEL).cpu()
        
        condensed_loader=torch.utils.data.DataLoader(TensorDataset(inverted_IMG, inverted_LABEL), batch_size=32, shuffle=True)

        ending_time = time.time()
        running_time.append(ending_time - starting_time)
        ipc_record.append(n_subclasses)

        testing_net=MLP(input_size=channel * im_size[0] * im_size[1], hidden_size=128, output_size=num_classes).to(device)
        optim_testing_net=torch.optim.Adam(testing_net.parameters(), lr=1e-3)
        for _ in range(30):
            for batch in condensed_loader:
                imgs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)
                optim_testing_net.zero_grad()
                output = testing_net(imgs)
                loss_test = criterion(output, labels)
                optim_testing_net.zero_grad()
                loss_test.backward()
                optim_testing_net.step()


        test_acc=test(testing_net, train_loader, device)
        print('Test accuracy: ', test_acc)
        print('Running time: ', running_time[-1])
        test_accs.append(test_acc)


    stat_data = {
        'Running Time': running_time,
        'Test Accuracy': test_accs,
        'IPC': ipc_record
    }

    df = pd.DataFrame(stat_data)


    df.to_csv('benchmark_proposed_fdm_gpu_batched_v1.csv', index=False)



if __name__ == '__main__':
    main()