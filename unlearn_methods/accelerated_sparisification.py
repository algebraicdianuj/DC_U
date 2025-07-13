
import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.distributions as dist
import copy


class Accelerated_Sparse_Unlearner:
    def __init__(self, 
                 original_model, 
                 retain_dataloader, 
                 forget_dataset, 
                 test_dataset, 
                 weight_distribution,
                 weight_sparsity,
                 k,
                 K,
                 device):
        
        self.original_model = original_model.to(device) 
        self.retain_dataloader = retain_dataloader
        self.forget_dataset = forget_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.batch_size=retain_dataloader.batch_size
        self.weight_distribution = weight_distribution
        self.k = k
        self.K = K
        self.weight_sparsity = weight_sparsity



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
        pairwise_dist = torch.cdist(x, y, p=2)
        kernel = torch.exp(-pairwise_dist**2 / (2 * sigma**2))
        return kernel
        
    
    def l1_regularization(self, model):
        """
        Calculate L1 norm of all model parameters - directly from author's code
        """
        params_vec = []
        for param in model.parameters():
            params_vec.append(param.view(-1))
        return torch.linalg.norm(torch.cat(params_vec), ord=1)
    

    


    def mmd_loss(self, losses_1, losses_2, sigma=1.0):
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
            
 

            

    def train_per_epoch(self, 
                    model, 
                    main_criterion,
                    loss_transformer,
                    optimizer,
                    retain_dataloader, 
                    forget_dataset, 
                    test_dataset, 
                    weight_distribution,
                    weight_sparsity
                    ):
        
        run_ce_loss = 0.0
        run_distribution_loss = 0.0
        run_sparsity_loss = 0.0

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
            output = model(data)
            loss = main_criterion(output, target)**2
            run_ce_loss += loss.item()

            test_output = model(test_data).detach()
            forget_output = model(forget_data)
            transformed_test_loss=loss_transformer(test_output, test_target)
            transformed_forget_loss=loss_transformer(forget_output, forget_target)
            distribution_loss=self.mmd_loss(transformed_test_loss, transformed_forget_loss)        
            loss += weight_distribution*distribution_loss
            run_distribution_loss += distribution_loss.item()
            
            sparsity_loss = self.l1_regularization(model)
            loss += weight_sparsity*sparsity_loss
            run_sparsity_loss += sparsity_loss.item()

            loss.backward()
            optimizer.step()


        return run_ce_loss/len(retain_dataloader), run_distribution_loss/len(retain_dataloader), run_sparsity_loss/len(retain_dataloader)






    def train(self, epochs=5, learning_rate=1e-4):

        current_model = self.original_model.to(self.device)


        optimizer=torch.optim.Adam(current_model.parameters(), lr=learning_rate)
        main_criterion = nn.CrossEntropyLoss(reduction='mean')  
        loss_transformer = self.QuantileCrossEntropyLoss(reduction='none', k=self.k, K=self.K)


        current_model.train()
        for epoch in range(epochs):
            run_ce_loss, run_distribution_loss, run_sparsity_loss = self.train_per_epoch(current_model, 
                                                                    main_criterion,
                                                                    loss_transformer,
                                                                    optimizer,
                                                                    self.retain_dataloader, 
                                                                    self.forget_dataset, 
                                                                    self.test_dataset, 
                                                                    self.weight_distribution,
                                                                    self.weight_sparsity
                                                                    )

                
        return current_model