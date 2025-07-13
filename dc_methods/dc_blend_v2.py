import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

from utils.utils import *
from utils.dc_losses import *


class Blend(nn.Module):
    def __init__(self, num_images, device='cpu'):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_images, device=device) / num_images)

    def forward(self, imgs):
        w = self.weights / self.weights.sum()
        w = w.view(-1, 1, 1, 1)
        return (imgs * w).sum(dim=0, keepdim=True)


def blend_DC(
    target_images,      # list of [n_i,C,H,W]
    target_labels,      # list of [n_i] 
    model,
    lr=0.05,
    num_iterations=500,
    device='cuda',
    batch_size=32,
):
    model.to(device).eval()            
    for p in model.parameters():
        p.requires_grad = False


    target_features = []
    with torch.no_grad():
        for imgs in target_images:
            all_feats = []
            for i in range(0, len(imgs), batch_size):
                batch = imgs[i:i+batch_size].to(device)
                f = model.embed(batch).detach().cpu()
                all_feats.append(f)
            target_features.append(torch.cat(all_feats, 0).to(device))

    target_images = [imgs.to(device) for imgs in target_images]


    blend_modules = [Blend(imgs.size(0), device).to(device)
                     for imgs in target_images]


    optimizer = torch.optim.SGD(
        [w for bm in blend_modules for w in bm.parameters()],
        lr=lr, momentum=0.5
    )

    num_targets = len(blend_modules)


    for it in range(num_iterations):
        optimizer.zero_grad()
        ref_loss = 0.0

        for imgs, blend, feat in zip(target_images, blend_modules, target_features):
            synth = blend(imgs)                      
            synth_feats = model.embed(synth)    
            loss_i = dist_match(synth_feats, feat) / num_targets
            ref_loss += loss_i.item()* num_targets
            loss_i.backward()                      

 
            del synth, synth_feats, loss_i

        print(f"Iteration {it+1}/{num_iterations}, Loss: {ref_loss}")

        optimizer.step()
        torch.cuda.empty_cache()


    syn_images = []
    syn_labels = []
    with torch.no_grad():
        for imgs, blend, lbls in zip(target_images, blend_modules, target_labels):
            final = blend(imgs).cpu()
            syn_images.append(final)
            syn_labels.append(torch.tensor([lbls[0].item()]))

    return torch.cat(syn_images, 0), torch.cat(syn_labels, 0)
