

import warnings
warnings.filterwarnings('ignore')
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.utils.data import Dataset
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
import copy
import random


def save_cond_imgs(condensed_images, condensed_labels, n_subclasses, channel, img_mean, img_std, label_map, name, exp, model_name):
    # Denormalize images
    image_syn_vis = copy.deepcopy(condensed_images.cpu())
    for ch in range(channel):
        image_syn_vis[:, ch] = image_syn_vis[:, ch] * img_std[ch] + img_mean[ch]
    image_syn_vis[image_syn_vis < 0] = 0.0
    image_syn_vis[image_syn_vis > 1] = 1.0

    # Get unique labels and organize images by class
    unique_labels = condensed_labels.unique().sort()[0]

    # Create a figure with separate columns for each class
    fig, axes = plt.subplots(n_subclasses, len(unique_labels), figsize=(len(unique_labels)*3, n_subclasses*3))

    # For single row case
    if n_subclasses == 1:
        axes = axes.reshape(1, -1)

    # Assign images to proper location in grid
    for class_idx, label in enumerate(unique_labels):
        # Get all images for this class
        class_indices = (condensed_labels == label).nonzero().squeeze()
        class_images = image_syn_vis[class_indices]
        
        # Add class label as column title (only on top row) with larger font
        axes[0, class_idx].set_title(label_map.get(label.item(), f"Class {label.item()}"), 
                                    fontsize=22)
        
        # Display each image in this class
        for i in range(min(n_subclasses, len(class_images))):
            if i < len(class_images):
                img = class_images[i].permute(1, 2, 0).numpy()
                axes[i, class_idx].imshow(img)
                axes[i, class_idx].axis('off')

    plt.tight_layout()
    plt.savefig(f"result/{name}_exp_{exp}_model_{model_name}.png")
    plt.close()



def save_cond_imgs_collage(condensed_images, condensed_labels, n_subclasses, channel, img_mean, img_std, label_map, name, exp, model_name):
    # 1. Denormalize images
    image_syn_vis = copy.deepcopy(condensed_images.cpu())
    for ch in range(channel):
        image_syn_vis[:, ch] = image_syn_vis[:, ch] * img_std[ch] + img_mean[ch]
    image_syn_vis = image_syn_vis.clamp(0.0, 1.0)

    # 2. Convert to PIL images
    pil_images = []
    for img_tensor in image_syn_vis:
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
        img_pil = Image.fromarray(img_np)
        pil_images.append(img_pil)

    # 3. Resize and prepare image list
    resized_images = []
    scale_range = (2.0, 3.0)
    for img in pil_images:
        scale = random.uniform(*scale_range)
        new_size = (int(img.width * scale), int(img.height * scale))
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS).convert("RGBA")
        resized_images.append(img_resized)

    # 4. Greedy blob placement
    canvas_size = (3000, 3000)
    canvas = Image.new("RGBA", canvas_size, (255, 255, 255, 0))
    occupied = []
    center_x, center_y = canvas_size[0] // 2, canvas_size[1] // 2

    for idx, img in enumerate(resized_images):
        w, h = img.size

        for _ in range(100):  # try up to 100 positions
            if idx == 0:
                x, y = center_x - w // 2, center_y - h // 2
            else:
                ref_x1, ref_y1, ref_x2, ref_y2 = random.choice(occupied)
                ref_x = (ref_x1 + ref_x2) // 2
                ref_y = (ref_y1 + ref_y2) // 2
                jitter = 100
                x = ref_x + random.randint(-jitter, jitter)
                y = ref_y + random.randint(-jitter, jitter)

            if 0 <= x < canvas_size[0] - w and 0 <= y < canvas_size[1] - h:
                canvas.alpha_composite(img, (x, y))
                occupied.append((x, y, x + w, y + h))
                break

    # 5. Crop unused space
    bbox = canvas.getbbox()
    cropped = canvas.crop(bbox)

    # 6. Save
    os.makedirs("result", exist_ok=True)
    out_path = f"result/{name}_exp_{exp}_model_{model_name}_compact_collage.png"
    cropped.convert("RGB").save(out_path)
    print(f"Saved greedy compact collage to {out_path}")
    cropped.show()



class TensorDatasett(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]
    



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




def save_data(result_directory_path, category, df, model_name, exp_num, unlearning_mode):

    file_path = os.path.join(
        result_directory_path, 
        f'{category}_stats_{model_name}_exp_{exp_num}_mode_{unlearning_mode}.csv'
    )
    df.to_csv(file_path, index=False)


def save_data2(result_directory_path, category, df, model_name, exp_num, ipc):

    file_path = os.path.join(
        result_directory_path, 
        f'{category}_stats_{model_name}_exp_{exp_num}_ipc_{ipc}.csv'
    )
    df.to_csv(file_path, index=False)


def save_data3(result_directory_path, category, df, model_name, exp_num, unlearning_mode, round_num):

    file_path = os.path.join(
        result_directory_path, 
        f'{category}_stats_{model_name}_exp_{exp_num}_mode_{unlearning_mode}_round_{round_num}.csv'
    )
    df.to_csv(file_path, index=False)




def train_model_with_raw_tensors(model, train_data, train_labels, epochs=100, lr=0.01, bs=128*2, device=torch.device('cuda')):
    dataset= TensorDatasett(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for batch in train_loader:
            img, label = batch
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
    return model




def sample_images(images, labels, target_label, N=100):
    indices = torch.where(labels == target_label)[0]
    sampled_indices = torch.randperm(len(indices))[:N]
    return images[sampled_indices]



def noisy_augment(normalized_img, K=5):

    if K <= 0:
        return normalized_img

    if normalized_img.dim() != 4 or normalized_img.size(0) != 1:
        raise ValueError(f"Expected input tensor of shape [1, C, H, W], but got {normalized_img.shape}")

    normalized_img_batch = normalized_img.repeat(K, 1, 1, 1)

    noise_mean = torch.empty(K, 1, 1, 1, device=normalized_img.device).uniform_(0, 0.1)
    noise_std = torch.empty(K, 1, 1, 1, device=normalized_img.device).uniform_(0, 0.1)

    noise = noise_mean + noise_std * torch.randn_like(normalized_img_batch)

    noisy_imgs = normalized_img_batch + noise

    noisy_imgs = torch.clamp(noisy_imgs, 0, 1)

    augmented_imgs = torch.cat([normalized_img, noisy_imgs], dim=0)

    return augmented_imgs


def get_data_to_be_condensed(fine_lab_idx_dictionary, train_images, train_labels, forget_indices, device):
        # Convert forget_indices to a set for O(1) lookup
    F_set = set(forget_indices)

    target_images = []
    target_labels = []

    # Process each array in indices_train_wrt_finelabels
    for i, (key, arr) in enumerate(fine_lab_idx_dictionary.items()):
        # Convert the array to a set for O(1) operations
        arr_set = set(arr)
        
        if F_set.intersection(arr_set):
            # If any elements of F are in the array, add all non-F elements to residual_indices
            res_indices=list(arr_set - F_set)
            target_images.append(train_images[res_indices].to(device))
            target_labels.append(train_labels[res_indices].to(device))

    return target_images, target_labels



def check_retain_size(retain_labels, forget_labels, retain_images, train_images, train_labels, img_batch_size):
    # Get unique forget labels
    unique_forget = torch.unique(forget_labels)
    
    # Check counts for each forget label in retain set
    for f_label in unique_forget:
        retain_count = torch.sum(retain_labels == f_label).item()
        if retain_count < img_batch_size:
            # If any class has fewer samples than needed, return full training set
            return train_images, train_labels
            
    return retain_images, retain_labels
