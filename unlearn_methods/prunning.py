import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import time
from torch.optim.lr_scheduler import MultiStepLR

def unlearn_with_pruning(model, retain_loader, 
                         unlearn_epochs=10, 
                         target_sparsity=0.95, 
                         learning_rate=0.01,
                         momentum=0.9, 
                         weight_decay=5e-4,
                         prune_step=2,
                         decreasing_lr="50,75",
                         print_freq=50,
                         device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    

    data_loaders = {"retain": retain_loader}
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )
    

    decreasing_lr = list(map(int, decreasing_lr.split(",")))
    
    scheduler = MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
    
    training_metrics = {
        'train_loss': [],
        'train_acc': [],
        'sparsity': []
    }
    
    initial_sparsity = check_sparsity(model)
    print(f"Initial model sparsity: {initial_sparsity:.2f}%")
    training_metrics['sparsity'].append(initial_sparsity)
    
    prune_rate = 1 - (1 - target_sparsity) ** (1 / ((unlearn_epochs - 1) // prune_step + 1))
    print(f"Progressive pruning rate per step: {prune_rate:.4f}")
    
    for epoch in range(unlearn_epochs):
        start_time = time.time()

        if (unlearn_epochs - epoch) % prune_step == 0:
            print(f"Epoch #{epoch}, applying L1 pruning")
            pruning_model(model, prune_rate)
            current_sparsity = check_sparsity(model)
            training_metrics['sparsity'].append(current_sparsity)
        

        train_loss, train_acc = FT_iter(
            data_loaders, model, criterion, optimizer, epoch, 
            {"print_freq": print_freq, "device": device}
        )
        
        training_metrics['train_loss'].append(train_loss)
        training_metrics['train_acc'].append(train_acc)
        
        scheduler.step()
        
        print(f"Epoch: [{epoch}] Learning rate: {optimizer.param_groups[0]['lr']}")
        print(f"One epoch duration: {time.time() - start_time:.2f}s")
    

    final_sparsity = check_sparsity(model)
    print(f"Final model sparsity: {final_sparsity:.2f}%")
    
    evaluation_result = {
        'training_metrics': training_metrics,
        'final_sparsity': final_sparsity
    }
    
    return model


def FT_iter(data_loaders, model, criterion, optimizer, epoch, args):

    train_loader = data_loaders["retain"]
    device = args.get("device", torch.device("cuda:0"))
    print_freq = args.get("print_freq", 50)
    

    model.train()
    
    total_loss = 0
    total_samples = 0
    
    start = time.time()
    
    for i, (image, target) in enumerate(train_loader):
        image = image.to(device)
        target = target.to(device)
        

        output = model(image)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        total_loss += loss.item() * image.size(0)
        total_samples += image.size(0)
        
        if (i + 1) % print_freq == 0:
            end = time.time()
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {3:.4f}\t"
                "Time {4:.2f}".format(
                    epoch, i, len(train_loader), 
                    loss.item(), end - start
                )
            )
            start = time.time()
    

    avg_loss = total_loss / total_samples
    placeholder_acc = 0.0
    
    return avg_loss, placeholder_acc


def pruning_model(model, px):

    print("Apply Unstructured L1 Pruning Globally (all conv layers)")
    parameters_to_prune = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m, "weight"))

    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )


def check_sparsity(model):

    sum_list = 0
    zero_sum = 0

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            sum_list = sum_list + float(m.weight.nelement())
            zero_sum = zero_sum + float(torch.sum(m.weight == 0))

    if zero_sum:
        remain_weight_ratio = 100 * (1 - zero_sum / sum_list)
        print("* remain weight ratio = {:.2f}%".format(remain_weight_ratio))
    else:
        print("no weight for calculating sparsity")
        remain_weight_ratio = 100.0

    return 100.0 - remain_weight_ratio  
