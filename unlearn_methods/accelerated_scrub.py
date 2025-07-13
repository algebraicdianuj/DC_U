
import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.distributions as dist
import copy





class Accelerated_SCRUB_Unlearner:
    def __init__(self, 
                 student_model,
                 teacher_model, 
                 retain_dataloader, 
                 forget_dataset, 
                 test_dataset, 
                 weight_distribution,
                 weight_gamma,
                 weight_beta,
                 kd_temp,
                 k,
                 K,
                 device):
        
        self.student_model = student_model.to(device) 
        self.teacher_model = teacher_model.to(device)
        self.retain_dataloader = retain_dataloader
        self.forget_dataset = forget_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.batch_size=retain_dataloader.batch_size
        self.weight_distribution = weight_distribution
        self.k = k
        self.K = K
        self.weight_gamma = weight_gamma
        self.weight_beta = weight_beta
        self.kd_temp = kd_temp



    def get_accuracy(self, model, dataloader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return 100 * correct / total





    def sample_data(self, dataset, num_samples, allowed_classes=None):
        # Get indices that satisfy the allowed_classes condition (if any)
        if allowed_classes is not None:
            # Build a list of indices whose label is in allowed_classes.
            # This assumes each dataset sample is a tuple (data, label) and label is a scalar.
            valid_indices = [i for i in range(len(dataset)) if dataset[i][1].item() in allowed_classes]
            # Shuffle valid indices and pick num_samples indices.
            indices = torch.tensor(valid_indices)[torch.randperm(len(valid_indices))][:num_samples]
        else:
            indices = torch.randperm(len(dataset))[:num_samples]
    
        # Stack the sampled data and labels
        data = torch.stack([dataset[i][0] for i in indices])
        labels = torch.stack([dataset[i][1] for i in indices])
        return data, labels
    


    def gaussian_kernel(self, x, y, sigma=1.0):
        # Compute the Gaussian kernel
        pairwise_dist = torch.cdist(x, y, p=2)
        kernel = torch.exp(-pairwise_dist**2 / (2 * sigma**2))
        return kernel
        
    



    def mmd_loss(self, losses_1, losses_2, sigma=1.0):
        # Compute the MMD loss
        x = losses_1.unsqueeze(-1)
        y = losses_2.unsqueeze(-1)
        kxx = self.gaussian_kernel(x, x, sigma).mean()
        kyy = self.gaussian_kernel(y, y, sigma).mean()
        kxy = self.gaussian_kernel(x, y, sigma).mean()
        return kxx + kyy - 2 * kxy
    




    class QuantileCrossEntropyLoss(nn.Module):
        def __init__(self, reduction='mean', k=10, K=50.0):

            super().__init__()
            self.reduction = reduction
            self.k = k
            self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
            self.K = K

        def forward(self, output, target):

          
            raw_losses = self.cross_entropy(output, target)  # shape: [batch_size]

            # Log transforming the cross-entropy losses
            raw_losses = torch.log1p(raw_losses + 1e-8)


            # Approximate the CDF using differentiable function
            # For each loss x_i, compute F(x_i) = mean_{j} sigmoid(k * (x_i - x_j))
            differences = raw_losses.unsqueeze(1) - raw_losses.unsqueeze(0)  # [batch_size, batch_size]
            differences = torch.relu(differences)
            sigmoid_diffs = torch.sigmoid(self.k * differences)  # [batch_size, batch_size]
            cdf_approx = sigmoid_diffs.mean(dim=1)  # [batch_size]

            # Apply inverse CDF (probit function) of standard normal distribution
            gaussian_losses = dist.Normal(0, 1).icdf(cdf_approx)

            # Clip the transformed losses to the range [-K, K]
            gaussian_losses = torch.clamp(gaussian_losses, -self.K, self.K)

    
            if self.reduction == 'mean':
                return torch.mean(gaussian_losses)
            elif self.reduction == 'sum':
                return torch.sum(gaussian_losses)
            else:
                return gaussian_losses
            

            
    class DistillKL(nn.Module):
        """KL divergence for distillation"""
        def __init__(self, T):
            super().__init__()
            self.T = T
    
        def forward(self, y_s, y_t):
            p_s = torch.nn.functional.log_softmax(y_s/self.T, dim=1)
            p_t = torch.nn.functional.softmax(y_t/self.T, dim=1)
            loss = torch.nn.functional.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
            return loss
    
                

    def train_forget(self, 
                    model_s,
                    model_t, 
                    criterion_div,
                    optimizer,
                    forget_dataset, 
                    ):
        
        model_s.train()
        model_t.eval()
        forget_dataloader = DataLoader(forget_dataset, batch_size=64, shuffle=True)
        
        for idx, (input, target) in enumerate(forget_dataloader):
            input = input.to(self.device)
            target = target.to(self.device)
    
            # Forward pass
            logit_s = model_s(input)
            with torch.no_grad():
                logit_t = model_t(input)
    
            # Calculate losses - maximize objective (forget knowledge)
            loss_div = criterion_div(logit_s, logit_t)
            loss = -0.2*loss_div  # Maximize divergence
            
    
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            


    def train_retain(
                    self, 
                    model_s, 
                    model_t,
                    main_criterion,
                    criterion_div,
                    loss_transformer,
                    optimizer,
                    retain_dataloader, 
                    forget_dataset, 
                    test_dataset, 
                    weight_distribution,
                    weight_gamma,
                    weight_beta
                    ):
        
        model_s.train()
        model_t.eval()
        for data, target in retain_dataloader:
            data, target = data.to(self.device), target.to(self.device)

            # Sample data from the 'forget_dataset'
            forget_data, forget_target = self.sample_data(forget_dataset, 4 * len(data))
            forget_data, forget_target = forget_data.to(self.device), forget_target.to(self.device)
            
            # Determine the unique classes in forget_target
            unique_classes = torch.unique(forget_target).tolist()
            
            # Now sample test data only from those classes in 'test_dataset'
            test_data, test_target = self.sample_data(test_dataset, 4 * len(data), allowed_classes=unique_classes)
            test_data, test_target = test_data.to(self.device), test_target.to(self.device)
            
            
            optimizer.zero_grad()
            output = model_s(data)
            with torch.no_grad():
                output_t = model_t(data)
            loss = weight_gamma * main_criterion(output, target)**2
            
            divergence_loss = criterion_div(output, output_t)
            loss += weight_beta * divergence_loss

            test_output   = model_s(test_data).detach()
            forget_output = model_s(forget_data)
            transformed_test_loss=loss_transformer(test_output, test_target)
            transformed_forget_loss=loss_transformer(forget_output, forget_target)
            distribution_loss=self.mmd_loss(transformed_test_loss, transformed_forget_loss)
            loss += weight_distribution*distribution_loss

            loss.backward()
            optimizer.step()






    def train(self, epochs=5, learning_rate=1e-4):

        model_s = self.student_model
        model_t = self.teacher_model
        
        optimizer=torch.optim.Adam(model_s.parameters(), lr=learning_rate)
        main_criterion = nn.CrossEntropyLoss(reduction='mean')  
        loss_transformer = self.QuantileCrossEntropyLoss(reduction='none', k=self.k, K=self.K)
        criterion_div = self.DistillKL(self.kd_temp)


        for epoch in range(epochs):
            self.train_forget(
                model_s,
                model_t,
                criterion_div,
                optimizer,
                self.forget_dataset, 
            )
            
            self.train_retain(
                model_s,
                model_t,
                main_criterion,
                criterion_div,
                loss_transformer,
                optimizer,
                self.retain_dataloader, 
                self.forget_dataset, 
                self.test_dataset, 
                self.weight_distribution,
                self.weight_gamma,
                self.weight_beta
                )

                
        return model_s