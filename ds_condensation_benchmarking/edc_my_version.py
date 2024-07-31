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
from math import ceil


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
    net.to(device)
    factor = max(1, int(np.sqrt(ipc)))
    decode_type = 'multi'
    max_size = 128   # Authors consider default 128
    #----------------------------------------------------------------------




    def subsample(data, target, max_size=-1):
        if (data.shape[0] > max_size) and (max_size > 0):
            indices = np.random.permutation(data.shape[0])
            data = data[indices[:max_size]]
            target = target[indices[:max_size]]

        return data, target

    def decode_zoom(img, target, factor, size=(32, 32)):
        """Uniform multi-formation
        """
        h = img.shape[-1]
        remained = h % factor
        if remained > 0:
            img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
        s_crop = ceil(h / factor)
        n_crop = factor**2

        cropped = []
        for i in range(factor):
            for j in range(factor):
                h_loc = i * s_crop
                w_loc = j * s_crop
                cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
        cropped = torch.cat(cropped)
        data_dec = nn.Upsample(size=size, mode='bilinear')(cropped)
        target_dec = torch.cat([target for _ in range(n_crop)])

        return data_dec, target_dec

    def decode_zoom_multi(img, target, factor_max, size=(32, 32)):
        """Multi-scale multi-formation
        """
        data_multi = []
        target_multi = []
        for factor in range(1, factor_max + 1):
            decoded = decode_zoom(img, target, factor, size)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

        return torch.cat(data_multi), torch.cat(target_multi)

    def decode_zoom_bound(img, target, factor_max, size=(32,32),bound=128):
        """Uniform multi-formation with bounded number of synthetic data
        """
        bound_cur = bound - len(img)
        budget = len(img)

        data_multi = []
        target_multi = []

        idx = 0
        decoded_total = 0
        for factor in range(factor_max, 0, -1):
            decode_size = factor**2
            if factor > 1:
                n = min(bound_cur // decode_size, budget)
            else:
                n = budget

            decoded = decode_zoom(img[idx:idx + n], target[idx:idx + n], factor, size)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

            idx += n
            budget -= n
            decoded_total += n * decode_size
            bound_cur = bound - decoded_total - budget

            if budget == 0:
                break

        data_multi = torch.cat(data_multi)
        target_multi = torch.cat(target_multi)
        return data_multi, target_multi

    def decode(data, target, factor, decode_type, size= (32,32), bound=128):
        """Multi-formation
        """
        if factor > 1:
            if decode_type == 'multi':
                data, target = decode_zoom_multi(data, target, factor, size)
            elif decode_type == 'bound':
                data, target = decode_zoom_bound(data, target, factor, size, bound=bound)
            else:
                data, target = decode_zoom(data, target, factor, size)

        return data, target
    

    def sample(S_images, S_targets, ipc, c, decode_type, size, max_size=128):
        """Sample synthetic data per class
        """
        idx_from = ipc * c
        idx_to = ipc * (c + 1)
        data = S_images[idx_from:idx_to]
        target = S_targets[idx_from:idx_to]

        data, target = decode(data, target, factor, decode_type, size, bound=max_size)
        data, target = subsample(data, target, max_size=max_size)
        return data, target
    


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


        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

       
        ''' initialize the synthetic data from random noise '''
        image_syn = torch.randn(size=(num_classes*ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=device)
        label_syn = torch.tensor([np.ones(ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        
        # ''' mixed initialization '''
        for c in range(num_classes):

            img = get_images(c, ipc*factor**2).detach().data

            s = im_size[0] // factor
            remained = im_size[0] % factor
            k = 0
            n = ipc

            h_loc = 0
            for i in range(factor):
                h_r = s + 1 if i < remained else s
                w_loc = 0
                for j in range(factor):
                    w_r = s + 1 if j < remained else s
                    img_part = F.interpolate(img[k * n:(k + 1) * n], size=(h_r, w_r))
                    image_syn.data[n * c:n * (c + 1), :, h_loc:h_loc + h_r,
                                    w_loc:w_loc + w_r] = img_part
                    w_loc += w_r
                    k += 1
                h_loc += h_r


        ''' training '''
        optimizer_img = torch.optim.Adam([image_syn, ], lr=lr_img) # optimizer_img for synthetic data
        optimizer_img.zero_grad()



        #---Starting the condensation process
        for it in range(condense_iterations):
            # Train Synthetic Data

            loss_syn = torch.tensor(0.0).to(device)
            for c in range(num_classes):
                img_real = get_images(c, batch_real)
                img_real=img_real.to(device)
                # img_syn = image_syn[c * ipc: (c + 1) * ipc].reshape((ipc, 3, 32, 32))
                img_syn,lab_syn = sample(image_syn, label_syn, ipc, c, decode_type, im_size, max_size=max_size)

                output_real = net.feature(img_real).detach()
                output_syn = net.feature(img_syn)

                loss_syn+= torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)


            optimizer_img.zero_grad()
            loss_syn.backward()
            optimizer_img.step()




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


    df.to_csv('benchmark_EDM_without_DSA.csv', index=False)




if __name__ == '__main__':
    main()