import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import rotate as scipyrotate
from torchvision import datasets, transforms
import random
from torch.utils.data import Dataset
import time
import pandas as pd

def main():
    batch_train=256
    lr_img=0.1   # authors consider default 1000
    lr_net=1e-3
    Iteration=1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 256
    num_classes = 10
    batch_real = 256
    channel = 3
    im_size = (32, 32)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform) # no augmentation
    student_steps=20
    max_start_epoch=25
    teacher_epoch=3
    teacher_steps=50
    lr_teacher=1e-2
    lr_student=1e-2
    ref_training_loader=torch.utils.data.DataLoader(dst_train, batch_size=batch_size, shuffle=True, num_workers=0)


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


    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten input
            x = F.relu(self.fc1(x))
            y= F.relu(self.fc2(x))
            z = self.fc3(y)
            return z



    def load_params(model, params):
        start_idx = 0
        for name, param in model.named_parameters():
            param_length = param.numel()
            param.data = params[start_idx:start_idx + param_length].view(param.shape)
            start_idx += param_length
        # #find the number of traininable parameters of model
        # print("=====================================================================================================")
        # print("Checking if parameters loaded are trainbale")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.numel())
        # print('>> Total trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
        # print('>> Total parameters:', sum(p.numel() for p in model.parameters()))
        # print("=====================================================================================================")



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

        if ipc == 1:
            outer_loop, inner_loop = 1, 1
        elif ipc == 10:
            outer_loop, inner_loop = 10, 50
        elif ipc == 50:
            outer_loop, inner_loop = 50, 10


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


        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=device)
        label_syn = torch.tensor([np.ones(ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]


        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(device)

        trajectory_teacher_params=[]
        teacher_model=MLP(input_size=channel * im_size[0] * im_size[1], hidden_size=32, output_size=num_classes).to(device)
        teacher_model_optimizer=torch.optim.SGD(teacher_model.parameters(), lr=lr_teacher)

        # recording training trajectory of teacher model
        for step in range(teacher_steps):
            c_loss=torch.tensor(0.0).to(device)
            for c in range(num_classes):
                img_real = get_images(c, batch_real)
                lab_real = torch.ones((img_real.shape[0],), device=device, dtype=torch.long) * c
                output_real = teacher_model(img_real)
                loss_real = nn.CrossEntropyLoss()(output_real, lab_real)
                c_loss += loss_real
    
            teacher_model_optimizer.zero_grad()
            c_loss.backward()
            teacher_model_optimizer.step()
            # trajectory_teacher_params.append([param.detach().clone() for param in teacher_model.parameters()])
            trajectory_teacher_params.append([torch.cat([p.data.to(device).reshape(-1).detach().clone() for p in teacher_model.parameters()], 0)])

        

        for it in range(Iteration):
        
            student_model = MLP(input_size=channel * im_size[0] * im_size[1], hidden_size=32, output_size=num_classes).to(device)
            student_model.train()

            # Get the starting and target parameters
            start_epoch= np.random.randint(0,max_start_epoch)
            starting_params = trajectory_teacher_params[start_epoch][0]
            target_params = trajectory_teacher_params[start_epoch+teacher_epoch][0]
            target_params = torch.cat([p.data.to(device).reshape(-1) for p in target_params], 0)
            #initialize student model with starting parameters
            trajectory_student_params = [torch.cat([p.data.to(device).reshape(-1) for p in student_model.parameters()], 0).requires_grad_(True)]
            starting_params = torch.cat([p.data.to(device).reshape(-1) for p in starting_params], 0)

            for step in range(student_steps):
                load_params(student_model, trajectory_student_params[-1])
                loss_c=torch.tensor(0.0).to(device)
                for c in range(num_classes):
                    img_syn = image_syn[c*ipc:(c+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))
                    lab_syn = torch.ones((ipc,), device=device, dtype=torch.long) * c

                    output_syn = student_model(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    loss_c += loss_syn

                grads = torch.autograd.grad(loss_c, student_model.parameters(), create_graph=True, allow_unused=False)
                grad_syn = torch.cat([g.view(-1) for g in grads])
                trajectory_student_params.append(trajectory_student_params[-1] - lr_student * grad_syn)


            param_loss = torch.nn.functional.mse_loss(trajectory_student_params[-1], target_params, reduction="sum")
            param_dist = torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
            param_loss /= param_dist
            grand_loss = param_loss
            optimizer_img.zero_grad()
            # grand_loss.backward(retain_graph=True)
            grand_loss.backward()
            optimizer_img.step()
            print('Iter %d, Loss: %.4f'%(it, grand_loss.item()))

            ending_time = time.time()
            running_time.append(ending_time - starting_time)
            ipc_record.append(ipc)


        image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
        dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
        trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=batch_train, shuffle=True, num_workers=0)

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


        test_acc=test(testing_net, ref_training_loader, device)
        test_accs.append(test_acc)


    stat_data = {
        'Running Time': running_time,
        'Test Accuracy': test_accs,
        'IPC': ipc_record
    }

    df = pd.DataFrame(stat_data)


    df.to_csv('benchmark_TM_without_DSA.csv', index=False)




if __name__ == '__main__':
    main()
