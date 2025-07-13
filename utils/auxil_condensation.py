import warnings
warnings.filterwarnings('ignore')
import torch




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

    # Generate noise: mean + std * random noise
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
    unique_forget = torch.unique(forget_labels)

    for f_label in unique_forget:
        retain_count = torch.sum(retain_labels == f_label).item()
        if retain_count < img_batch_size:
            return train_images, train_labels
            
    return retain_images, retain_labels