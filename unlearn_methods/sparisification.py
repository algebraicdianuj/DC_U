import torch
import torch.nn as nn
import copy
import time

def unlearn_with_l1_sparsity(model, 
                            retain_loader,
                            test_loader=None,
                            unlearn_epochs=10, 
                            learning_rate=0.01,
                            momentum=0.9, 
                            weight_decay=5e-4,
                            alpha=5e-4,
                            no_l1_epochs=2,
                            warmup=0,
                            decreasing_lr="50,75",
                            print_freq=50,
                            device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model = model.to(device)
    
    initialization = copy.deepcopy(model.state_dict())

    data_loaders = {"retain": retain_loader}
    if test_loader is not None:
        data_loaders["test"] = test_loader
    

    criterion = nn.CrossEntropyLoss().to(device)
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )
    
    decreasing_lr = list(map(int, decreasing_lr.split(",")))
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )
    
    training_metrics = {
        'l1_norm': []
    }
    
    for epoch in range(unlearn_epochs):
        start_time = time.time()
        
        if epoch < unlearn_epochs - no_l1_epochs:
            current_alpha = alpha * (1 - epoch / (unlearn_epochs - no_l1_epochs))
        else:
            current_alpha = 0
        
        for i, (image, target) in enumerate(retain_loader):

            if epoch < warmup:
                warmup_lr(epoch, i + 1, optimizer, one_epoch_step=len(retain_loader))
            
            image = image.to(device)
            target = target.to(device)
            

            output = model(image)
            loss = criterion(output, target)

            if current_alpha > 0:
                l1_norm = l1_regularization(model)
                loss += current_alpha * l1_norm
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        

        current_l1_norm = l1_regularization(model).item()
        training_metrics['l1_norm'].append(current_l1_norm)

        scheduler.step()
        
        print(f"Epoch: [{epoch}] Learning rate: {optimizer.param_groups[0]['lr']}")
        print(f"L1 norm: {current_l1_norm:.4f}")
        print(f"One epoch duration: {time.time() - start_time:.2f}s")
    
    final_sparsity = check_sparsity(model)
    
 
    evaluation_result = {
        'training_metrics': training_metrics,
        'final_sparsity': final_sparsity,
        'initialization': initialization
    }
    
    return model


def l1_regularization(model):

    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def check_sparsity(model):

    sum_list = 0
    zero_sum = 0

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            sum_list = sum_list + float(m.weight.nelement())
            zero_sum = zero_sum + float(torch.sum(m.weight == 0))

    if zero_sum:
        remain_weight_ratio = 100 * (1 - zero_sum / sum_list)
        print(f"* remain weight ratio = {remain_weight_ratio:.2f}%")
        return 100.0 - remain_weight_ratio  # Return sparsity percentage
    else:
        print("no weight for calculating sparsity")
        return 0.0


def warmup_lr(epoch, step, optimizer, one_epoch_step, warmup_epoch=5):

    if epoch < warmup_epoch:
        epoch = epoch * one_epoch_step + step
        epoch_total = warmup_epoch * one_epoch_step
        lr = 0.1 * (epoch / epoch_total)
    else:
        lr = 0.1
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
