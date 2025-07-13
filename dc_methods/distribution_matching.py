import os
import time
import copy
import argparse
import numpy as np
import torch
from torchvision.utils import save_image
from utils.dc_utils import (
    get_loops, get_dataset, get_network, get_time,
    DiffAugment, ParamDiffAug
)
from utils.loading_model import get_model


def DM_condensation(model_fn, 
                     images_all, 
                     labels_all, 
                     num_iterations,
                     ipc,
                     lr_img,
                     channel, 
                     im_size, 
                     num_classes,
                     bn,
                     do_dsa,
                     dsa_strategy,
                     dsa_param,
                     batch_real,
                     mean, 
                     std,
                     device
                     ):

    # Build per-class indices
    indices_class = [[] for _ in range(num_classes)]
    for idx, lab in enumerate(labels_all.tolist()):
        indices_class[lab].append(idx)

    # Helper to sample real images by class
    def get_images(c, n):
        perm = np.random.permutation(indices_class[c])[:n]
        return images_all[perm]

    # Initialize synthetic data
    image_syn = torch.randn(
        (num_classes * ipc, channel, im_size[0], im_size[1]),
        device=device, requires_grad=True
    )
    label_syn = torch.tensor(
        [np.ones(ipc) * i for i in range(num_classes)],
        dtype=torch.long, device=device
    ).view(-1)


    for c in range(num_classes):
        image_syn.data[c * ipc:(c + 1) * ipc] = \
            get_images(c, ipc).detach().data


    optimizer_img = torch.optim.SGD([image_syn], lr=lr_img, momentum=0.5)
    data_save_exp = []

    print(f'{get_time()} training begins')
    for it in range(num_iterations):
        # Training step for synthetic data
        net = model_fn().to(device)
        net.train()
        for p in net.parameters():
            p.requires_grad = False
        embed = net.module.embed if hasattr(net, 'module') else net.embed

        loss = torch.tensor(0.0, device=device)
        if not bn:
            for c in range(num_classes):
                img_real = get_images(c, batch_real)
                img_syn_c = image_syn[c * ipc:(c + 1) * ipc]
                if do_dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, dsa_strategy, seed, dsa_param)
                    img_syn_c = DiffAugment(img_syn_c, dsa_strategy, seed, dsa_param)
                out_r = embed(img_real.to(device)).detach()
                out_s = embed(img_syn_c)
                loss += ((out_r.mean(0) - out_s.mean(0)) ** 2).sum()
        else:
            real_list, syn_list = [], []
            for c in range(num_classes):
                img_real = get_images(c, batch_real)
                img_syn_c = image_syn[c * ipc:(c + 1) * ipc]
                if do_dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, dsa_strategy, seed, dsa_param)
                    img_syn_c = DiffAugment(img_syn_c, dsa_strategy, seed, dsa_param)
                real_list.append(img_real)
                syn_list.append(img_syn_c)
            out_r = embed(torch.cat(real_list, 0).to(device)).detach()
            out_s = embed(torch.cat(syn_list, 0))
            loss += ((out_r.view(num_classes, batch_real, -1).mean(1)
                      - out_s.view(num_classes, ipc, -1).mean(1)) ** 2).sum()

        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()


    image_syn_final = image_syn.detach().cpu().clone()
    label_syn_final = label_syn.detach().cpu().clone()

    return image_syn_final, label_syn_final
