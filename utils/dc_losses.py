import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn

def dist_match(feature_syn, feature_tar):
    return nn.MSELoss()(torch.mean(feature_syn, dim=0), torch.mean(feature_tar, dim=0))


def feature_complexify_1(features):
    return torch.mean(torch.var(features, dim=1))


def feature_complexify_2(features):
    std_synth = torch.std(features, dim=1).unsqueeze(0)
    if std_synth.shape[1] > 1:
        std_of_std_synth = torch.std(std_synth, dim=1)
    else:
        std_of_std_synth = std_synth

    std_of_std_synth = std_of_std_synth.squeeze()
    return std_of_std_synth


def logit_complexify(logits):
    return torch.mean(torch.var(logits, dim=1))

