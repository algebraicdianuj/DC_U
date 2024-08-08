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
import pandas as pd


def main():
    # Define your network architecture (MLP)
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size*4)
            self.fc2 = nn.Linear(hidden_size*4, hidden_size*2)
            self.fc3 = nn.Linear(hidden_size*2, hidden_size)
            self.fc4 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten input
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            y = F.relu(self.fc3(x))
            z = self.fc4(y)
            return z
        
        def feature(self, x):
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            y = F.relu(self.fc3(x))
            return y

    batch_size = 256
    num_classes = 10
    batch_real = 256
    ipc = 10   # according to authors, recommended outer_loop, inner_loop = 10, 50
    channel = 3
    im_size = (32, 32)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform) # no augmentation
    train_loader = torch.utils.data.DataLoader(dst_train, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    im_size = (32, 32)
    channel = 3
    #------------------Train the Net--------------------------------


    net= MLP(input_size=channel * im_size[0] * im_size[1], hidden_size=128, output_size=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(net.parameters(), lr=1e-3)

    # Teacher training before starting the dataset distillation process
    for epochy in range(30):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = net(data) 
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    del loss
    del output

    # make all parameters non-trainable, so as to make image_syn the only trainable parameter
    for param in list(net.parameters()):
        param.requires_grad = False
    #---------------------------------------------------------------------------
    net.eval()

    #--------Hyperparameters-----------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    condense_iterations=20000 #Authors consider default 20000
    num_classes = 10
    batch_real = 256
    channel = 3
    im_size = (32, 32)
    lr_img = 1e-1  # Authors consider default 1.0
    lr_net=1e-2   # Authors consider default 1e-2
    net.to(device)
    model_type='MLP'
    net_generate_interval=30     # Authors consider default 30, and is the network update interval
    net_push_num=1        # Authors consider default 1, and is the number of networks to be pushed
    net_num= 100         # Authors consider default 100, and is the number of networks to be maintained
    net_begin=0          # Authors consider default 0, and is the index of the first network to be used
    net_end=100000          # Authors consider default 100000, and is the index of the last network to be used
    train_net_num=2     # Authors consider default 2, and is the number of networks to be used for training
    embed_last=-1
    fetch_net_num=2
    model_train_steps=10   # Authors consider default 10, and is the number of training steps for each network
    trained_bs=256   # Authors consider default 256, and is the batch size for training networks
    #----------------------------------------------------------------------


    def downscale(image_syn, scale_factor):
        image_syn = F.upsample(image_syn, scale_factor=scale_factor, mode='bilinear')
        return image_syn
    

    def get_network(model_type, channel, num_classes, im_size):
        if model_type == 'MLP':
            net=MLP(input_size=channel * im_size[0] * im_size[1], hidden_size=128, output_size=num_classes).to(device)

        else:
            raise ValueError('Not implemented')

        return net



    def number_sign_augment(image_syn, label_syn):
        half_length = image_syn.shape[2]//2
        # import pdb; pdb.set_trace()
        a, b, c, d = image_syn[:, :, :half_length, :half_length].clone(), image_syn[:, :, half_length:, :half_length].clone(), image_syn[:, :, :half_length, half_length:].clone(), image_syn[:, :, half_length:, half_length:].clone()
        a, b, c, d = F.upsample(a, scale_factor=2, mode='bilinear'), F.upsample(b, scale_factor=2, mode='bilinear'), \
            F.upsample(c, scale_factor=2, mode='bilinear'), F.upsample(d, scale_factor=2, mode='bilinear')
        # a, b, c, d = image_syn.clone(), image_syn.clone(), image_syn.clone(), image_syn.clone()
        image_syn_augmented = torch.concat([a, b, c, d], dim=0)
        label_syn_augmented = label_syn.repeat(4)
        return image_syn_augmented, label_syn_augmented

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
        

    class TensorDataset(Dataset):
        def __init__(self, images, labels): # images: n x c x h x w tensor
            self.images = images.detach().float()
            self.labels = labels.detach()

        def __getitem__(self, index):
            return self.images[index], self.labels[index]

        def __len__(self):
            return self.images.shape[0]
        


    def epoch(mode, dataloader, net, optimizer, criterion):
        net = net.to(device)
        criterion = criterion.to(device)

        if mode == 'train':
            net.train()
        else:
            net.eval()

        for i_batch, datum in enumerate(dataloader):
            img = datum[0].float().to(device)
            lab = datum[1].long().to(device)

            output = net(img)
            loss = criterion(output, lab)
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()



    running_time=[]
    test_accs=[]
    ipc_record=[]

    ipc_trys=[1,10,50]

    for exp in range(len(ipc_trys)):
        print('\n================== Exp %d ==================\n '%exp)

        ipc=ipc_trys[exp]
        starting_time = time.time()

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=device)


        def get_images(c=None, n=0): # get random n images from class c
            if c is not None:
                idx_shuffle = np.random.permutation(indices_class[c])[:n]
                return images_all[idx_shuffle]
            else:
                assert n > 0, 'n must be larger than 0'
                indices_flat = [_ for sublist in indices_class for _ in sublist]
                idx_shuffle = np.random.permutation(indices_flat)[:n]
                return images_all[idx_shuffle], labels_all[idx_shuffle]
       
        ''' initialize the synthetic data from random noise '''
        image_syn = torch.randn(size=(num_classes*ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=device)
        label_syn = torch.tensor([np.ones(ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        
        # ''' copy the real data to synthetic data for initialization'''
        for c in range(num_classes):
            image_syn.data[c*ipc:(c+1)*ipc] = get_images(c, ipc).detach().data
            half_size = im_size[0]//2
            image_syn.data[c*ipc:(c+1)*ipc, :, :half_size, :half_size] = downscale(get_images(c, ipc), 0.5).detach().data
            image_syn.data[c*ipc:(c+1)*ipc, :, half_size:, :half_size] = downscale(get_images(c, ipc), 0.5).detach().data
            image_syn.data[c*ipc:(c+1)*ipc, :, :half_size, half_size:] = downscale(get_images(c, ipc), 0.5).detach().data
            image_syn.data[c*ipc:(c+1)*ipc, :, half_size:, half_size:] = downscale(get_images(c, ipc), 0.5).detach().data


        ''' training '''
        optimizer_img = torch.optim.Adam([image_syn, ], lr=lr_img) # optimizer_img for synthetic data
        optimizer_img.zero_grad()


        net_list = list()
        optimizer_list = list()
        for net_index in range(3):
            net = get_network(model_type, channel, num_classes, im_size).to(device) # get a random model
            net.train()
            optimizer_net = torch.optim.SGD(net.parameters(), lr=lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            net_list.append(net)
            optimizer_list.append(optimizer_net)

        # =======================================
        #---Starting the condensation process
        for it in range(condense_iterations):

            if it % net_generate_interval == 0:
                # append and pop net list:
                for _ in range(net_push_num):
                    if len(net_list) == net_num:
                        net_list.pop(0)
                        optimizer_list.pop(0)
                    net = get_network(model_type, channel, num_classes, im_size).to(device) # get a random model
                    net.train()
                    optimizer_net = torch.optim.SGD(net.parameters(), lr=lr_net)  # optimizer_img for synthetic data
                    optimizer_net.zero_grad()
                    net_list.append(net)
                    optimizer_list.append(optimizer_net)


            _ = list(range(len(net_list)))
            if len(_[net_begin: net_end]) > 10:
                _ = _[net_begin: net_end]
            random.shuffle(_)
            net_index_list = _[:train_net_num]

            train_net_list = [net_list[ind] for ind in net_index_list]

            embed_list = [net.feature for net in train_net_list]


            # Train Synthetic Data
            # loss_syn = torch.tensor(0.0).to(device)

            for net_ind in range(len(train_net_list)):
                net = train_net_list[net_ind]
                net.eval()
                embed = embed_list[net_ind]
  
                for c in range(num_classes):
                    loss_syn = torch.tensor(0.0).to(device)
                    img_real = get_images(c, batch_real)
                    img_real=img_real.to(device)
                    img_syn = image_syn[c * ipc: (c + 1) * ipc].reshape((ipc, 3, 32, 32))
                    lab_syn=label_syn[c * ipc: (c + 1) * ipc]

                    # aug thing
                    img_syn, lab_syn = number_sign_augment(img_syn, lab_syn)


                    output_real = embed(img_real).detach()
                    output_syn = embed(img_syn)

                    loss_syn+= torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)
                    logits_syn = net(img_syn)
                    # print('Iter: %d, Net: %d, Class: %d, Loss: %.4f, Acc: %.4f' % (it, net_ind, c, loss_syn.item(), test(net, train_loader, device)))
                    

                    # syn_ce_loss = 0
                    # if syn_ce:
                    #     weight_i = net_acc.value()[0] if net_acc.n != 0 else 0
                    #     syn_ce_loss += (F.cross_entropy(logits_syn, lab_syn.repeat(aug_num)) * weight_i)

                    #     loss_c += (syn_ce_loss * ce_weight)



                    optimizer_img.zero_grad()
                    loss_syn.backward()
                    optimizer_img.step()

                    image_syn.data = torch.clamp(image_syn.data, min=0., max=1.)

            
            shuffled_net_index = list(range(len(net_list)))
            random.shuffle(shuffled_net_index)
            for j in range(min(fetch_net_num, len(shuffled_net_index))):
                training_net_idx = shuffled_net_index[j]
                net_train = net_list[training_net_idx]
                net_train.train()
                optimizer_net_train = optimizer_list[training_net_idx]
                for i in range(model_train_steps):
                    img_real_, lab_real_ = get_images(c=None, n=trained_bs)
                    real_logit = net_train(img_real_)
                    syn_cls_loss = criterion(real_logit, lab_real_)
                    optimizer_net_train.zero_grad()
                    syn_cls_loss.backward()
                    optimizer_net_train.step()



        ending_time = time.time()
        running_time.append(ending_time - starting_time)
        ipc_record.append(ipc)

        image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
        dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
        trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=32, shuffle=True, num_workers=0)

        testing_net=MLP(input_size=channel * im_size[0] * im_size[1], hidden_size=128, output_size=num_classes).to(device)
        optim_testing_net=torch.optim.Adam(testing_net.parameters(), lr=1e-3)
        for _ in range(30):
            for batch in trainloader:
                imgs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)
                optim_testing_net.zero_grad()
                output = testing_net(imgs)
                loss_test = criterion(output, labels)
                optim_testing_net.zero_grad()
                loss_test.backward()
                optim_testing_net.step()


        test_acc=test(testing_net, train_loader, device)
        test_accs.append(test_acc)


    stat_data = {
        'Running Time': running_time,
        'Test Accuracy': test_accs,
        'IPC': ipc_record
    }

    df = pd.DataFrame(stat_data)


    df.to_csv('benchmark_IDM_without_DSA.csv', index=False)




if __name__ == '__main__':
    main()
