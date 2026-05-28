from tqdm import trange, tqdm
import torch
import torch.nn.functional as F
from optimizers import (
    FedAvgOptimizer,
    FedProxOptimizer,
    FedDANEOptimizer,
    FedSGDOptimizer,
    FedNovaOptimizer,
)
import numpy as np
import logging

"""
Transform data from using two wearables to one at a time
"""
class RemoveSensor(torch.utils.data.Dataset):
    def __init__(self, dataset, selection):
        self.dataset = dataset
        self.selection = selection

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, y = self.dataset[idx]
        X = X.clone()

        if self.selection[idx] == 0:
            X[:,0:9] = 0
        else:
            X[:,9:18] = 0

        return X, y

"""
Instance of client device performing local model training
"""
class Client:
    """Encapsulates a single client’s local training logic, with progress bars."""
    def __init__(self, model, train_loader, config, device, client_id, wearable_selection = None):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.client_id = client_id

        # Select one wearable sensor based on energy usage
        if self.config.hetero_wearables:
            dataset = train_loader.dataset
            modified_dataset = RemoveSensor(dataset, wearable_selection)

            self.train_loader = torch.utils.data.DataLoader(
                modified_dataset,
                batch_size=train_loader.batch_size,
                shuffle=True,
                num_workers=train_loader.num_workers,
                pin_memory=train_loader.pin_memory,
            )
        
        else:
            self.train_loader = train_loader
        
        # Map algorithm name to optimizer class
        optimizer_map = {
            "fedavg":  FedAvgOptimizer,
            "fedprox": FedProxOptimizer,
            "feddane": FedDANEOptimizer,
            "fedsgd":  FedSGDOptimizer,
            "fednova": FedNovaOptimizer
        }
        optimizer_cls = optimizer_map[config.algorithm]

        # Instantiate optimizer (pass mu for prox/DANE)
        if config.algorithm in ("fedprox", "feddane"):
            self.optimizer = optimizer_cls(
                self.model.parameters(),
                lr=config.local_lr,
                mu=config.global_lr
            )
        elif config.algorithm in ("fedavg", "fednova"):
            self.optimizer = optimizer_cls(
                self.model.parameters(),
                lr=config.local_lr,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optimizer_cls(
                self.model.parameters(),
                lr=config.local_lr
            )

    def train(self, global_model=None, global_grads=None):
        """
        Perform local model training

        Args:
            global_model: copy of the global model
            global_grads: used by FedSGD and FedDANE

        Returns:
            self.model: copy of model parameters after local training
            step_count: number of steps taken by model during local training
            total_loss.item(): total loss for client during local training
        """
        step_count = 0
        squared_losses = []
        self.model.train()
        epochs = 1 if self.config.algorithm == "fedsgd" else self.config.local_epochs

        for epoch in trange(1, epochs + 1,
                            desc=f"Client {self.client_id} Epoch",
                            leave=False):
            # iterate over batches with tqdm
            for batch_idx, (data, target) in enumerate(
                    tqdm(self.train_loader,
                         desc=f"Client {self.client_id} Batches",
                         leave=False,
                         unit="batch")):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                logits, _ = self.model(data)
                loss = F.cross_entropy(logits, target)
                squared_losses.append((loss ** 2).detach().cpu())
                loss.backward()

                # call step with correct signature
                if self.config.algorithm == "fedavg":
                    self.optimizer.step()
                elif self.config.algorithm == "fedprox":
                    self.optimizer.step(global_params=global_model)
                elif self.config.algorithm == "feddane":
                    self.optimizer.step(global_params=global_model, global_gradients=global_grads)
                elif self.config.algorithm == "fedsgd":
                    self.optimizer.zero_grad()
                    self.optimizer.step(global_gradients=global_grads)
                elif self.config.algorithm == "fednova":
                    self.optimizer.step()
                    step_count += 1
        
        num_samples = len(squared_losses)
        rms_losses = np.sqrt(np.mean(squared_losses))
        total_loss = num_samples * rms_losses
        
        return self.model, step_count, total_loss.item()