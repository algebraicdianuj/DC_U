

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
import torch.nn.utils.prune as prune
from model.model import *



class TensorDatasett(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]
    


class CombinedModel(nn.Module):
    def __init__(self, databank, final):
        super(CombinedModel, self).__init__()
        self.databank = databank
        self.final = final

    def forward(self, x):
        x = self.databank(x)
        # print(x.shape)
        x = self.final(x)
        return x
    
    def feature(self, x):
        x = self.databank(x)
        return x




class Databank(nn.Module):
    def __init__(self,beggining,intermediate):
        super(Databank, self).__init__()
        self.beggining=beggining
        self.intermediate=intermediate

    def forward(self,x):
        x=self.beggining(x)
        x=self.intermediate(x)
        return x

    def hidden(self,x):
        x=self.beggining(x)
        return x


# Defining a new module for weighted average
class WeightedAverage(nn.Module):
    def __init__(self, num_batches):
        super(WeightedAverage, self).__init__()
        self.weights = nn.Parameter(1/num_batches*torch.ones(num_batches, ))
        # self.fc1 = nn.Linear(128, 128)

    def forward(self, imgs):
        imgs = imgs.view(imgs.shape[0], -1)
        weighted_imgs = imgs * self.weights.view(-1, 1)
        weighted_imgs = torch.sum(weighted_imgs, dim=0, keepdim=True)
        # pro_weighted_avg = F.relu(self.fc1(weighted_avg))
        weighted_imgs = weighted_imgs.reshape(1, 3, 32, 32)
        return weighted_imgs


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



def get_images_from_testloader(num_images, class_label, test_loader):
    sampled_images = [] 
    
    # Loop over batches
    for images, labels in test_loader:
        indices = (labels == class_label).nonzero(as_tuple=True)[0]
        

        for index in indices:
            sampled_images.append(images[index])
            

            if len(sampled_images) >= num_images:
                return torch.stack(sampled_images)[:num_images] 

    if len(sampled_images) < num_images:
        print(f"Warning: Only found {len(sampled_images)} images of class {class_label}.")
        return torch.stack(sampled_images) 
    


def extract_features(model, dataloader, device):
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            feature = model.feature(data)
            feature = feature.view(feature.size(0), -1)  # Flatten spatial dimensions
            features.append(feature.cpu())
            labels.append(label)
    
    return torch.cat(features, 0), torch.cat(labels, 0)



def create_sub_classes(tensor, labels, model, num_classes=10, sub_divisions=10, device=torch.device('cpu')):
    new_labels = torch.zeros_like(labels)
    original_labels_dict = {}
    
    model.to(device)
    
    # Create a DataLoader to facilitate feature extraction
    dataset = TensorDatasett(tensor, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Extract features
    features, _ = extract_features(model, loader, device)
    print("Extracting Features Done!")
    
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





# Function for L1 regularization
def l1_regularization(model):
    params_vec = [param.view(-1) for param in model.parameters()]
    return torch.linalg.norm(torch.cat(params_vec), ord=1)



def param_dist(model, swa_model, p):
    #This is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
    dist = 0.
    for p1, p2 in zip(model.parameters(), swa_model.parameters()):
        dist += torch.norm(p1 - p2, p='fro')
    return p * dist


#this is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
#For SGDA smoothing

def avg_fn(averaged_model_parameter, model_parameter, beta):
    return (1 - beta) * averaged_model_parameter + beta * model_parameter



class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss
    


def pruning_model(model, px):
    print("Apply Unstructured L1 Pruning Globally")
    parameters_to_prune = []
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:  # check if the module has a weight parameter
            parameters_to_prune.append((module, "weight"))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )




def find_elbow_layer_index(model, dataloader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model.to('cpu')

    # Get gradients
    for inputs, labels in dataloader:
        inputs.requires_grad = True
        outputs = model.original_model(inputs)  # Access the original_model attribute
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)
        break  # Only one batch is needed

    # Store layer gradient norms
    gradient_norms = []
    for name, param in model.original_model.named_parameters():  # Access parameters of the original_model
        grad_norm = param.grad.norm().item()
        gradient_norms.append(grad_norm)

    # Function to find elbow point in a curve
    def find_elbow_point(values):
        n_points = len(values)
        all_coords = np.vstack((range(n_points), values)).T
        first_point = all_coords[0]
        line_vector = all_coords[-1] - all_coords[0]
        line_vector_norm = line_vector / np.sqrt(np.sum(line_vector**2))

        vector_from_first = all_coords - first_point
        dot_product = np.dot(vector_from_first, line_vector_norm)
        proj_vector_from_first = np.outer(dot_product, line_vector_norm)
        vector_to_line = vector_from_first - proj_vector_from_first

        dist_to_line = np.sqrt(np.sum(vector_to_line**2, axis=1))
        elbow_index = np.argmax(dist_to_line)
        return elbow_index
    
    # Find and return elbow point index
    elbow_index = find_elbow_point(gradient_norms)
    return elbow_index

