import warnings
warnings.filterwarnings('ignore')
import torch
from utils.utils import *
from torch.utils.data import DataLoader


def extract_features(model, dataloader, device):
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            # feature = model(data)
            feature = model.embed(data)
            feature = feature.view(feature.size(0), -1)  # Flatten spatial dimensions
            features.append(feature.cpu())
            labels.append(label)
    
    return torch.cat(features, 0), torch.cat(labels, 0)




# def kmeans_pytorch(X, num_clusters, num_iterations=100, tol=1e-4):
#     N, D = X.shape
    
#     # Randomly initialize cluster centers
#     C = X[torch.randperm(N)[:num_clusters]]
    
#     for i in range(num_iterations):
#         # Compute distances
#         distances = torch.cdist(X, C)
        
#         # Assign points to nearest cluster
#         labels = torch.argmin(distances, dim=1)
        
#         # Update cluster centers
#         new_C = torch.stack([X[labels == k].mean(dim=0) for k in range(num_clusters)])
        
#         # Check for convergence
#         if torch.abs(new_C - C).sum() < tol:
#             break
        
#         C = new_C
    
#     return labels
    

import torch

def kmeans_pytorch(X, num_clusters, num_iterations=100, tol=1e-4):
    """
    K-means with k-means++ init and empty-cluster re-seeding.
    X:       (N, D) tensor of features
    returns: labels (N,) with values in [0, num_clusters)
    """
    N, D = X.shape
    device = X.device

    # --- 1) k-means++ initialization ---
    centroids = torch.empty((num_clusters, D), device=device)
    # 1.1 pick first centroid uniformly
    first_idx = torch.randint(0, N, (1,), device=device)
    centroids[0] = X[first_idx]

    # keep track of min squared distances to any chosen centroid
    closest_dist_sq = torch.full((N,), float('inf'), device=device)

    for c in range(1, num_clusters):
        # update distance to nearest existing centroid
        dist_to_new = torch.sum((X - centroids[c-1])**2, dim=1)
        closest_dist_sq = torch.minimum(closest_dist_sq, dist_to_new)

        # sample next centroid proportional to squared distance
        probs = closest_dist_sq / torch.sum(closest_dist_sq)
        next_idx = torch.multinomial(probs, 1)
        centroids[c] = X[next_idx]

    # --- 2) standard k-means loop ---
    for it in range(num_iterations):
        # 2.1 assign labels
        # (N, num_clusters)
        dists = torch.cdist(X, centroids, p=2)
        labels = torch.argmin(dists, dim=1)

        # 2.2 compute new centroids, with handling for empty clusters
        new_centroids = torch.zeros_like(centroids)
        for k in range(num_clusters):
            mask = (labels == k)
            if mask.any():
                new_centroids[k] = X[mask].mean(dim=0)
            else:
                # empty cluster: re-seed to a random point
                rand_idx = torch.randint(0, N, (1,), device=device)
                new_centroids[k] = X[rand_idx]

        # 2.3 check for convergence
        shift = torch.norm(new_centroids - centroids, dim=1).sum()
        centroids = new_centroids
        if shift < tol:
            break

    return labels




def create_sub_classes(inputs, 
                       labels, 
                       model, 
                       num_classes=10, 
                       sub_divisions=10,
                       device=torch.device('cuda')):

    # 1) Prepare
    new_labels = torch.zeros_like(labels)
    total_fine = num_classes * sub_divisions

    # pre-populate every possible fine-label with an empty list
    indices_train_wrt_finelabels = {fine: [] for fine in range(total_fine)}

    model.to(device)
    dataset = TensorDatasett(inputs, labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    features_all, _ = extract_features(model, loader, device)

    # 2) Run K-means per coarse class
    for i in range(num_classes):
        mask = (labels == i)
        class_feats = features_all[mask]

        # get sub_divisions cluster‐IDs in [0, sub_divisions)
        class_new_labels = kmeans_pytorch(class_feats, num_clusters=sub_divisions)

        # offset them to global fine‐label range
        new_subclass_labels = i * sub_divisions + class_new_labels
        new_labels[mask] = new_subclass_labels

    # 3) Build indices dict over **all** fine-label IDs
    for fine in range(total_fine):
        idxs = torch.where(new_labels == fine)[0]
        # if idxs is empty, .tolist() is []
        indices_train_wrt_finelabels[fine] = idxs.tolist()

    # 4) Compute subclass_info only for non-empty clusters
    subclass_info = {}
 

    return new_labels, indices_train_wrt_finelabels, subclass_info





def dataset_sampling(
                     indices_train_wrt_finelabels, 
                     train_images, 
                     train_labels,
                     ):
    


    
    target_images = []
    target_labels = []
    
    # Process each array in indices_train_wrt_finelabels
    for i, (_,arr) in enumerate(indices_train_wrt_finelabels.items()):
        # Convert the array to a set for O(1) operations
        arr_set = set(arr)
        

        target_images.append(train_images[list(arr)])
        target_labels.append(train_labels[list(arr)])


    
    return target_images, target_labels





def seperated_dataset_sampling(
                     indices_train_wrt_finelabels, 
                     train_images, 
                     train_labels,
                     forget_indices
                     ):
    
    F_set = set(forget_indices)


    target_images = []
    target_labels = []

    residual_indices = []

    # Process each array in indices_train_wrt_finelabels
    for i, (_,arr) in enumerate(indices_train_wrt_finelabels.items()):
        # Convert the array to a set for O(1) operations
        arr_set = set(arr)
        
        if not F_set.intersection(arr_set):
            target_images.append(train_images[list(arr)])
            target_labels.append(train_labels[list(arr)])

        else:
            res_arr = arr_set - F_set
            residual_indices.extend(res_arr)

    residual_indices = list(residual_indices)
    residual_images = train_images[residual_indices]
    residual_labels = train_labels[residual_indices]

    return target_images, target_labels, residual_images, residual_labels







def condensation_sampling(
                     indices_train_wrt_finelabels, 
                     train_images, 
                     train_labels
                     ):
    
    
    target_images = []
    target_labels = []
    sub_labels = []

    # Process each array in indices_train_wrt_finelabels
    for i, (sub_lab,arr) in enumerate(indices_train_wrt_finelabels.items()):

        target_images.append(train_images[list(arr)])
        target_labels.append(train_labels[list(arr)])
        sub_labels.append(sub_lab)


    return target_images, target_labels, sub_labels










def seperated_sampling_v2(
                     indices_train_wrt_finelabels, 
                     train_images, 
                     train_labels,
                     forget_indices,
                     sub_labels,
                     condensed_images,
                     condensed_labels
                     ):
    
    F_set = set(forget_indices)

    residual_indices = []
    cond_imgs = []
    cond_labels = []

    # Process each array in indices_train_wrt_finelabels
    for i, (sub_lab,arr) in enumerate(indices_train_wrt_finelabels.items()):
        # Convert the array to a set for O(1) operations
        arr_set = set(arr)
        
        if F_set.intersection(arr_set):
            res_arr = arr_set - F_set
            residual_indices.extend(res_arr)

        else:
            idx = sub_labels.index(sub_lab)
            cond_imgs.append(condensed_images[idx].unsqueeze(0))
            cond_labels.append(condensed_labels[idx].unsqueeze(0))

    residual_indices = list(residual_indices)
    residual_images = train_images[residual_indices]
    residual_labels = train_labels[residual_indices]

    cond_images = torch.cat(cond_imgs, dim=0)
    cond_labels = torch.cat(cond_labels, dim=0)

    return cond_images, cond_labels, residual_images, residual_labels


