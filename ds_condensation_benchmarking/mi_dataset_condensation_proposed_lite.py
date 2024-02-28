
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


    # Define your network architecture (MLP)
    class InvertedMLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(InvertedMLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size*4)
            self.fc3 = nn.Linear(hidden_size*4, hidden_size*16)
            self.fc4 = nn.Linear(hidden_size*16, hidden_size*32)
            self.fc5 = nn.Linear(hidden_size*32, output_size[0]*output_size[1]*output_size[2])
            self.output_size = output_size


        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x= F.relu(self.fc3(x))
            x= F.relu(self.fc4(x))
            y= self.fc5(x)
            y= y.reshape(y.shape[0],self.output_size[0],self.output_size[1],self.output_size[2])
            return y

    class InverterDataset(Dataset):
        def __init__(self, tensor1, tensor2, tensor3):
            assert tensor1.size(0) == tensor2.size(0) == tensor3.size(0)  # Ensure tensors have the same length
            self.tensor1 = tensor1
            self.tensor2 = tensor2
            self.tensor3 = tensor3

        def __len__(self):
            return self.tensor1.size(0)  # Return the total number of samples

        def __getitem__(self, idx):
            sample_tensor1 = self.tensor1[idx]
            sample_tensor2 = self.tensor2[idx]
            sample_tensor3 = self.tensor3[idx]
            return sample_tensor1, sample_tensor2, sample_tensor3



    class Inverter_Net(nn.Module):
        def __init__(self,beggining,end):
            super(Inverter_Net, self).__init__()
            self.beggining=beggining
            self.end=end

        def forward(self, x):
            x = self.beggining(x)
            x = self.end(x)
            return x



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
    img_shape=(channel,im_size[0],im_size[1])

    for exp in range(len(number_subclasses)):
        print('\n================== Exp %d ==================\n '%exp)
        starting_time = time.time()
        n_subclasses=number_subclasses[exp]
        new_lab_train, original_labels_dict = create_sub_classes(training_images, training_labels, model=net, num_classes=num_classes, sub_divisions=n_subclasses)
        soft_labels = torch.nn.functional.one_hot(new_lab_train.cpu(), num_classes=num_classes*n_subclasses).float()
        inversion_dataset=InverterDataset(soft_labels,training_labels,training_images)

        net.to('cpu')

        inversion_loader=torch.utils.data.DataLoader(inversion_dataset, batch_size=batch_size, shuffle=True)

        inverted_net=InvertedMLP(input_size=num_classes*n_subclasses, hidden_size=hidden_size, output_size=img_shape)

        combined_model=Inverter_Net(inverted_net,net).to(device)

        for param in combined_model.parameters():
            param.requires_grad = False

        for param in combined_model.beggining.parameters():
            param.requires_grad = True


        optim_combo=torch.optim.Adam(combined_model.parameters(), lr=1e-3)
        total_ep=200
        for _ in range(total_ep):
            run_loss=0.0
            for batch_idx, (data, target, img) in enumerate(inversion_loader):
                data, target, img = data.to(device), target.to(device), img.to(device)
                inter_img=combined_model.beggining(data)
                output = combined_model.end(inter_img) 
                n_loss = nn.CrossEntropyLoss()(output, target)
                n_loss+= nn.MSELoss()(inter_img,img)
                optim_combo.zero_grad()
                n_loss.backward()
                optim_combo.step()
                run_loss+=n_loss.item()


        estim_imges=[]
        estim_class=[]
        inverter=combined_model.beggining
        for idx in range(n_subclasses*num_classes):
            # create one hot vector of size 10
            classer=idx
            one_hot = torch.zeros((1, n_subclasses*num_classes))
            one_hot[0, classer] = 1

            estim_img=inverter(one_hot.to(device)).detach().cpu()
            estim_imges.append(estim_img)
            estim_class.append(original_labels_dict[idx])

        condensed_images=torch.cat(estim_imges, dim=0)
        condensed_labels=torch.tensor(estim_class, dtype=torch.long, device=device)

        condensed_dataset=TensorDataset(condensed_images, condensed_labels)
        condensed_loader=DataLoader(condensed_dataset, batch_size=256, shuffle=True)
        

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


    df.to_csv('benchmark_proposed_mi_lite.csv', index=False)



if __name__ == '__main__':
    main()