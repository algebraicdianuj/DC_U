import torch
import torch.nn as nn
import time
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR



class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = torch.nn.functional.log_softmax(y_s/self.T, dim=1)
        p_t = torch.nn.functional.softmax(y_t/self.T, dim=1)
        loss = torch.nn.functional.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        return loss


def train_retain(epoch, retain_loader, model_s, model_t, criterion_list, optimizer, 
                gamma, beta, device, print_freq=12):

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    
    
    model_s.train()
    model_t.eval()


    for idx, (input, target) in enumerate(retain_loader):
        input = input.to(device)
        target = target.to(device)
  
        # Forward pass
        logit_s = model_s(input)
        with torch.no_grad():
            logit_t = model_t(input)

        # Calculate losses - minimize objective (retain knowledge)
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)
        loss = gamma * loss_cls + beta * loss_div
        

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_forget(epoch, forget_loader, model_s, model_t, criterion_list, optimizer, 
                device, print_freq=12):

    criterion_div = criterion_list[1]
    
    
    model_s.train()
    model_t.eval()

 
    for idx, (input, target) in enumerate(forget_loader):
        input = input.to(device)
        target = target.to(device)


        logit_s = model_s(input)
        with torch.no_grad():
            logit_t = model_t(input)

        loss_div = criterion_div(logit_s, logit_t)
        loss = -0.2*loss_div 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def scrub_model(
    teacher,
    student,
    retain_loader,
    forget_loader,
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    warmup = 2,
    m_steps=1,
    epochs=10,
    kd_temp=4.0,
    gamma=0.1,
    beta=1.0,
    milestones=[5, 10, 15],
    device='cuda'
):

    teacher_model = teacher
    teacher_model.to(device)
    teacher_model.eval()
    

    student_model = student.to(device)

    optimizer = torch.optim.SGD(
        student_model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )


    lambda0 = lambda cur_iter: (cur_iter + 1) / warmup if cur_iter < warmup else (
        0.5 * (1.0 + np.cos(np.pi * ((cur_iter - warmup) / (epochs - warmup))))
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(kd_temp)
    criterion_kd = DistillKL(kd_temp)
    
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)
    criterion_list.append(criterion_div)
    criterion_list.append(criterion_kd)
    
 
    
    print(f"Retain loader length: {len(retain_loader)}, Forget loader length: {len(forget_loader)}")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        print(f"Epoch #{epoch}, Learning rate: {optimizer.param_groups[0]['lr']}")
        
        if epoch <= m_steps:
            train_forget(
                epoch, forget_loader, student_model, teacher_model, criterion_list, 
                optimizer, device
            )
        
        train_retain(
            epoch, retain_loader, student_model, teacher_model, criterion_list, 
            optimizer, gamma, beta, device
        )
        
        scheduler.step()
        

    return student_model
