import os
import random
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils.dc_utils_v2 import (
    get_network, get_eval_pool, evaluate_synset,
    get_time, epoch, DiffAugment, ParamDiffAug,
    number_sign_augment, parser_bool, downscale
)
from utils.loading_model import get_model

import torch.nn.functional as F


def IDM_condensation(
    model_fns,
    images_all,
    labels_all,
    ipc,
    channel,
    im_size,
    num_classes,
    lr_img,
    num_iterations,
    do_dsa,
    do_aug,
    dsa_strategy,
    dsa_param,
    batch_real,
    aug_num,
    ce_weight,
    syn_ce,
    mean,
    std,
    device
):

    indices_class = [[] for _ in range(num_classes)]
    for idx, lab in enumerate(labels_all.tolist()):
        indices_class[lab].append(idx)

    def get_images(c=None, n=0):
        if c is not None:
            idxs = np.random.permutation(indices_class[c])[:n]
            return images_all[idxs]
        else:
            flat = [j for cls in indices_class for j in cls]
            idxs = np.random.permutation(flat)[:n]
            return images_all[idxs], labels_all[idxs]


    image_syn = torch.randn(
        (num_classes * ipc, channel, *im_size),
        device=device, requires_grad=True
    )


    label_syn = torch.tensor(
        [np.ones(ipc) * c for c in range(num_classes)],
        dtype=torch.long, device=device
    ).view(-1)


    for c in range(num_classes):
        patch = get_images(c, ipc).detach()
        if not do_aug:
            image_syn.data[c*ipc:(c+1)*ipc] = patch
        else:
            half = im_size[0] // 2
            for i in range(4):
                block = downscale(patch, 0.5).detach()
                x0 = half * (i // 2)
                y0 = half * (i % 2)
                image_syn.data[
                    c*ipc:(c+1)*ipc, :, x0:x0+half, y0:y0+half
                ] = block

    # Set up optimizer
    optim_img = torch.optim.SGD([image_syn], lr=lr_img, momentum=0.5)

    # Instantiate multiple nets
    nets = [model_fn().to(device) for model_fn in model_fns]

    # Optimization loop
    for it in range(num_iterations):
        loss_total = torch.tensor(0.0, device=device)

        # Compute loss across classes and nets
        for c in range(num_classes):
            img_real = get_images(c, batch_real).to(device)
            img_syn = image_syn[c*ipc:(c+1)*ipc]
            lab_syn = label_syn[c*ipc:(c+1)*ipc]

            if do_aug:
                img_syn, lab_syn = number_sign_augment(img_syn, lab_syn)
            if do_dsa:
                seed = int(time.time() * 1e3) % 100000
                img_real = DiffAugment(img_real, dsa_strategy, seed, dsa_param)
                img_syn = DiffAugment(img_syn, dsa_strategy, seed, dsa_param)

            for net in nets:
                net.eval()
                fr = net.embed(img_real).detach()
                fs = net.embed(img_syn)
                loss_fm = ((fr.mean(0) - fs.mean(0))**2).sum()
                logits = net(img_syn)
                loss_ce = F.cross_entropy(logits, lab_syn.repeat(aug_num))
                if syn_ce:
                    loss_ce = loss_ce * ce_weight
                loss_total = loss_total + loss_fm + loss_ce

 
        optim_img.zero_grad()
        loss_total.backward()
        optim_img.step()

    image_syn_eval, label_syn_eval =image_syn.detach().cpu(), label_syn.detach().cpu()

    return image_syn_eval, label_syn_eval

