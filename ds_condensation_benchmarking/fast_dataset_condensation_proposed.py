
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


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    im_size = (32, 32)
    channel = 3

    #------------------Train the Net--------------------------------
    net= MLP(input_size=channel * im_size[0] * im_size[1], hidden_size=hidden_size, output_size=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
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

    net.eval()




    # Defining a new module for weighted average
    class WeightedAverage(nn.Module):
        def __init__(self, num_batches):
            super(WeightedAverage, self).__init__()
            self.weights = nn.Parameter(1/num_batches*torch.ones(num_batches, device=device))
            # self.fc1 = nn.Linear(128, 128)

        def forward(self, imgs):
            imgs = imgs.view(imgs.shape[0], -1)
            weighted_imgs = imgs * self.weights.view(-1, 1)
            weighted_imgs = torch.sum(weighted_imgs, dim=0, keepdim=True)
            # pro_weighted_avg = F.relu(self.fc1(weighted_avg))
            weighted_imgs = weighted_imgs.reshape(1, 3, 32, 32)
            return weighted_imgs




    def Average(ref_imgs_all, pretrained=net, num_epochs=100):

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


    def create_sub_classes(tensor, labels, model, num_classes=10, sub_divisions=10):
        new_labels = torch.zeros_like(labels)
        original_labels_dict = {}
        
        # Load the pretrained model for feature extraction
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Create a DataLoader to facilitate feature extraction
        dataset = TensorDataset(tensor, labels)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        # Extract features
        features, _ = extract_features(model, loader, device)
        
        for i in range(num_classes):
            mask = labels == i
            class_features = features[mask]
            
            # Apply k-means clustering
            kmeans = KMeans(n_clusters=sub_divisions).fit(class_features)
            class_new_labels = torch.tensor(kmeans.labels_, dtype=torch.long)
            
            # Assign new labels
            new_subclass_labels = i * sub_divisions + class_new_labels
            new_labels[mask] = new_subclass_labels

            # Store original label reference
            for j in range(sub_divisions):
                original_labels_dict[int(i * sub_divisions + j)] = i
        
        return new_labels, original_labels_dict




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
        print('\n================== Exp %d ==================\n '%exp)
        starting_time = time.time()
        n_subclasses=number_subclasses[exp]
        new_lab_train, original_labels_dict = create_sub_classes(training_images, training_labels, model=net, num_classes=num_classes, sub_divisions=n_subclasses)

        img_shape=(1,3,32,32)  # Shape of the image
        inverted_IMG=[]
        inverted_LABEL=[]
        indices_train_wrt_bucket=[]
        bucket_labbies=torch.unique(new_lab_train).tolist()

        for idx in bucket_labbies:

            indices_idx = torch.where(new_lab_train.to(device)==idx)[0]

            indices_train_wrt_bucket.append(indices_idx.cpu())


            ref_imgs_all = training_images[indices_idx.cpu()]
            ref_labs_all= training_labels[indices_idx.cpu()]

            inverted_image = Average(ref_imgs_all, pretrained=net, num_epochs=100)

            inverted_IMG.append(inverted_image)
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
        test_accs.append(test_acc)


    stat_data = {
        'Running Time': running_time,
        'Test Accuracy': test_accs,
        'IPC': ipc_record
    }

    df = pd.DataFrame(stat_data)


    df.to_csv('benchmark_proposed.csv', index=False)



if __name__ == '__main__':
    main()