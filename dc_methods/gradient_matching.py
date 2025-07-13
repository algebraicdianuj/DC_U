import os
import time
import copy
import argparse
import numpy as np
import torch
from torchvision.utils import save_image
from utils.dc_utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
from utils.loading_model import get_model

def get_loops(ipc):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if ipc <= 5:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    else:
        outer_loop, inner_loop = 0, 0
        exit('loop hyper-parameters are not defined for %d ipc'%ipc)
    return outer_loop, inner_loop


def GM_condensation(model_fn, 
                     images_all, 
                     labels_all, 
                     criterion,
                     num_iterations,
                     ipc,
                     lr_img,
                     lr_net,
                     channel, 
                     im_size, 
                     num_classes,
                     dis_metric,
                     bn,
                     do_dsa,
                     dsa_strategy,
                     dsa_param,
                     batch_real,
                     batch_train,
                     mean, 
                     std,
                     device
                     ):
    """
    Run synthetic-data optimization (DC/DSA) for one experiment.
    Returns final synthetic images and labels.
    """

    outer_loop, inner_loop = get_loops(ipc)
    print(f'{get_time()} outer-loop={outer_loop}, inner-loop={inner_loop}')


    # build per-class indices
    indices_class = [[] for _ in range(num_classes)]
    for idx, lab in enumerate(labels_all.tolist()):
        indices_class[lab].append(idx)

    def get_images(c, n):
        idxs = np.random.permutation(indices_class[c])[:n]
        return images_all[idxs]

    # initialize synthetic data
    image_syn = torch.randn(
        (num_classes * ipc, channel, im_size[0], im_size[1]),
        device=device, requires_grad=True
    )
    label_syn = torch.tensor(
        [np.ones(ipc) * i for i in range(num_classes)],
        dtype=torch.long, device=device
    ).view(-1)


    for c in range(num_classes):
        image_syn.data[c*ipc:(c+1)*ipc] = \
            get_images(c, ipc).detach().data

    optimizer_img = torch.optim.SGD([image_syn], lr=lr_img, momentum=0.5)
 
    print(f'{get_time()} start optimizing synthetic data')
    for it in range(num_iterations + 1):

        # outer-loop data optimization
        net = model_fn().to(device)
        net.train()
        optimizer_net = torch.optim.SGD(net.parameters(), lr=lr_net)

        loss_total = 0.0
        for ol in range(outer_loop):
            # freeze BN stats
            if any('BatchNorm' in m._get_name() for m in net.modules()):
                bn_real = torch.cat([get_images(c, 16) for c in range(num_classes)], 0)
                net.train(); _ = net(bn_real.to(device))
                for m in net.modules():
                    if 'BatchNorm' in m._get_name(): m.eval()

            # compute gradient matching loss
            loss = torch.tensor(0.0, device=device)
            net_params = list(net.parameters())
            for c in range(num_classes):
                real = get_images(c, batch_real)
                lab_r = torch.full((real.size(0),), c,
                                    dtype=torch.long, device=device)
                syn = image_syn[c*ipc:(c+1)*ipc]
                lab_s = torch.full((ipc,), c,
                                    dtype=torch.long, device=device)

                if do_dsa:
                    seed = int(time.time()*1e3)%100000
                    real = DiffAugment(real, dsa_strategy, seed, dsa_param)
                    syn = DiffAugment(syn, dsa_strategy, seed, dsa_param)

                out_r = net(real.to(device))
                gr = torch.autograd.grad(criterion(out_r, lab_r),
                                         net_params, retain_graph=False)
                out_s = net(syn)
                gs = torch.autograd.grad(criterion(out_s, lab_s),
                                         net_params, create_graph=True)
                loss += match_loss(gs, gr, dis_metric, device)

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_total += loss.item()

            # optional inner-loop fine-tune
            if ol < outer_loop - 1:
                syn_data = TensorDataset(
                    image_syn.detach().clone(), label_syn.detach().clone())
                loader = torch.utils.data.DataLoader(
                    syn_data, batch_size=batch_train,
                    shuffle=True)
                

                for il in range(inner_loop):
                    epoch('train', loader, net,
                          optimizer_net, criterion,
                          do_dsa, dsa_strategy, dsa_param, device)

        if it % 10 == 0:
            avg = loss_total/(num_classes*outer_loop)
            print(f'{get_time()} iter={it}, loss={avg:.4f}')

  
    return image_syn.detach().cpu(), label_syn.detach().cpu()
