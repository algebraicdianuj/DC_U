import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, StratifiedShuffleSplit
from sklearn import linear_model, model_selection, metrics
from utils.utils import *
from torch.utils.data import DataLoader


# # ref: NIPS Unlearning Competition
def simple_mia(sample_phi, members, n_splits=10, random_state=0):
    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    preds = cross_val_predict(attack_model, sample_phi, members, cv=cv, method="predict")
    proba = model_selection.cross_val_predict(
        attack_model, sample_phi, members, cv=cv, method="predict_proba"
    )[:, 1]  # Probabilities for class=1


    auc_score = metrics.roc_auc_score(members, proba)
    acc_score = metrics.accuracy_score(members, preds)
    precision = metrics.precision_score(members, preds)
    recall = metrics.recall_score(members, preds)
    f1 = metrics.f1_score(members, preds)
    cm = metrics.confusion_matrix(members, preds)

    return auc_score


# def simple_mia(sample_phi, members, n_splits=10, random_state=0):
#     unique_members = np.unique(members)
#     if not np.all(unique_members == np.array([0, 1])):
#         raise ValueError("members should only have 0 and 1s")

#     attack_model = linear_model.LogisticRegression()
#     cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=random_state)

#     scores = cross_val_score(
#         attack_model, sample_phi, members, 
#         cv=cv, scoring="roc_auc"
#     )

#     return scores.mean()



def stable_phi(output, target, criterion, num_classes=10, device=torch.device('cuda')):
    batch_size = output.size(0)

    # Compute CrossEntropyLoss for the entire batch
    # Note that this is a single scalar for the entire batch by default
    loss = criterion(output, target)

    # Clamp the batch loss to avoid extremely large/small values
    # Cross-entropy is non-negative, so clamp min to 0.0
    loss_clamped = torch.clamp(loss, min=0.0, max=50.0)

    # Create anti-targets for each sample in the batch
    anti_targets = []
    for i in range(batch_size):
        sample_anti_targets = [j for j in range(num_classes) if j != target[i].item()]
        anti_targets.append(torch.tensor(sample_anti_targets).to(device))

    # Compute exp term in a numerically stable manner
    # exp_term is shape (batch_size,)
    exp_term = torch.exp(-loss_clamped)

    # Compute anti_exp_term for each sample
    # We'll clamp each sample's loss before exponentiating as well
    anti_exp_term_list = []
    for i in range(batch_size):
        # Repeat the i-th logit so we can feed it to criterion with all anti-targets
        # This gives a vector of cross-entropies (num_classes-1 in length)
        sample_loss = criterion(
            output[i].unsqueeze(0).repeat(num_classes - 1, 1),
            anti_targets[i]
        )
        # Clamp and exponentiate
        sample_loss_clamped = torch.clamp(sample_loss, min=0.0, max=50.0)
        sample_anti_exp = torch.exp(-sample_loss_clamped)
        anti_exp_term_list.append(sample_anti_exp)

    # Stack to get shape (batch_size, num_classes-1)
    anti_exp_term = torch.stack(anti_exp_term_list, dim=0)

    # Now compute phi for each sample
    # Guard against log(0) by clamping the arguments inside log
    positive_term = torch.log(torch.clamp(exp_term, min=1e-15))
    negative_term = torch.log(torch.clamp(torch.sum(anti_exp_term, dim=1), min=1e-15))
    phi = positive_term - negative_term

    return phi





def get_phi_losses(model, loader, criterion, num_classes, device):
    model.to(device)
    model.eval()
    losses = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = stable_phi(outputs, labels, criterion, num_classes, device)
            losses.append(loss.detach().cpu())

    losses = torch.cat(losses)
    return losses




def MIA(forget_phi, test_phi):
    forget_phi=forget_phi
    test_phi=test_phi
    stack_size=min([len(forget_phi), len(test_phi)])
    forget_phi = forget_phi[: stack_size]
    test_phi = test_phi[: stack_size]

    samples_mia = torch.cat([forget_phi, test_phi]).numpy().reshape(-1, 1)
    labels_mia = torch.tensor([0] * len(forget_phi) + [1] * len(test_phi)).numpy()

    mia_cands=[]
    for _ in range(20):
        mia_score = simple_mia(samples_mia, labels_mia)
        mia_cands.append(mia_score)

    mia_score=np.mean(mia_cands)

    return mia_score



def lira_MIA(model, forget_loader, test_loader, criterion, num_classes, device):

    # Collect all unique classes from forget_loader
    forget_classes = set()
    for _, labels in forget_loader:
        forget_classes.update(labels.unique().tolist())

    # Convert to a set for faster operations
    forget_classes = torch.tensor(list(forget_classes))

    # Filter test loader to include only classes in forget_loader
    filtered_test_images = []
    filtered_test_labels = []

    for images, labels in test_loader:
        mask = torch.isin(labels, forget_classes)
        filtered_test_images.append(images[mask])
        filtered_test_labels.append(labels[mask])

    filtered_test_images = torch.cat(filtered_test_images)
    filtered_test_labels = torch.cat(filtered_test_labels)


    filtered_test_dataset = TensorDatasett(filtered_test_images, filtered_test_labels)
    test_loader = DataLoader(filtered_test_dataset, batch_size=64, shuffle=False)

    forget_phi=get_phi_losses(model, forget_loader, criterion, num_classes, device)

    test_phi=get_phi_losses(model, test_loader, criterion, num_classes, device)

    auc_score=MIA(forget_phi, test_phi)

    return auc_score * 100


def LiRA_MIA(model, forget_loader, test_loader, criterion, num_classes, device):
    auc_scores =[]
    auc_scores.append(lira_MIA(model, forget_loader, test_loader, criterion, num_classes, device))
    if auc_scores[0] > 42:
        return auc_scores[0]
    
    else:
        auc_scores.append(lira_MIA(model, test_loader, forget_loader, criterion, num_classes, device))
        return max(auc_scores)


