"""
MIT License

Copyright (c) 2026 Farhad Rezazadeh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from tqdm import trange, tqdm
import torch
import torch.nn.functional as F
from aggregators import (
    FedAvgOptimizer,
    FedProxOptimizer,
    FedDANEOptimizer,
    FedSGDOptimizer,
    FedNovaOptimizer,
    FedPerOptimizer,
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
        
        # Map algorithm name to aggregator class
        aggregator_map = {
            "fedavg":  FedAvgOptimizer,
            "fedprox": FedProxOptimizer,
            "feddane": FedDANEOptimizer,
            "fedsgd":  FedSGDOptimizer,
            "fednova": FedNovaOptimizer,
            "fedper": FedPerOptimizer
        }
        agg_mechanism = aggregator_map[config.algorithm]

        # Select aggregation mechanism
        if config.algorithm in ("fedprox", "feddane"):
            self.aggregator = agg_mechanism(
                self.model.parameters(),
                lr=config.local_lr,
                mu=config.global_lr
            )
        elif config.algorithm in ("fedavg", "fednova", "fedper"):
            self.aggregator = agg_mechanism(
                self.model.parameters(),
                lr=config.local_lr,
                weight_decay=config.weight_decay
            )
        else:
            self.aggregator = agg_mechanism(
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
        entropy_scores = []
        self.model.train()

        epochs = 1 if self.config.algorithm == "fedsgd" else self.config.local_epochs

        # Train client models
        for epoch in trange(1, epochs + 1,
                            desc=f"Client {self.client_id} Epoch",
                            leave=False):

            for batch_idx, (data, target) in enumerate(
                    tqdm(self.train_loader,
                         desc=f"Client {self.client_id} Batches",
                         leave=False,
                         unit="batch")):
                data, target = data.to(self.device), target.to(self.device)
                self.aggregator.zero_grad()
                logits, _ = self.model(data)
                loss = F.cross_entropy(logits, target)
                squared_losses.append((loss ** 2).detach().cpu())
                loss.backward()

                # Entropy
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                sample_entropy = -(probs * log_probs).sum(dim=-1)
                mean_entropy = sample_entropy.mean()
                entropy_scores.append((mean_entropy ** 2).detach().cpu())

                # call step with correct signature
                if self.config.algorithm in ("fedavg", "fedper"):
                    self.aggregator.step()
                elif self.config.algorithm == "fedprox":
                    self.aggregator.step(global_params=global_model)
                elif self.config.algorithm == "feddane":
                    self.aggregator.step(global_params=global_model, global_gradients=global_grads)
                elif self.config.algorithm == "fedsgd":
                    self.aggregator.zero_grad()
                    self.aggregator.step(global_gradients=global_grads)
                elif self.config.algorithm == "fednova":
                    self.aggregator.step()
                    step_count += 1
        
        num_samples = len(squared_losses)
        rms_losses = np.sqrt(np.mean(squared_losses))
        total_loss = num_samples * rms_losses
        total_entropy = num_samples * np.sqrt(np.mean(entropy_scores))
        
        return self.model, step_count, total_loss.item(), total_entropy.item()
