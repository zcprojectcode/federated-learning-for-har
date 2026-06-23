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

import copy
import torch
import math
import numpy as np
import torch.nn as nn
from model import InertialTransformer
from client import Client
from client_energy import ClientEnergy
from wearable_energy import calculate_wearable_energy
from client_eval import evaluate_local
from utils import calculate_federated_score, calculate_util, compute_global_grads
import logging
import random

class Server:
    def __init__(self, config, data_loader, device, SEED):
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.config = config
        self.data_loader = data_loader
        self.device = device
        self.wearable_selection = None
        self.window_counts = data_loader.get_windows()
        self.client_energies = []
        self.participated = []
        device_types = []

        # Randomly assign clients a device profile
        for _ in range(config.num_clients):
            device_types.append(random.choices(["H", "M", "L"], k = config.devices_per_client))
        
        # Track client energies
        for cli in range(config.num_clients):
            energies = [ClientEnergy(i) for i in device_types[cli]]
            self.client_energies.append(energies)

        # Track client utilisation and accuracy
        self.client_utils = [0 for i in range(config.num_clients)]
        self.local_accuracies = [0 for i in range(config.num_clients)]

        # Track client scores (accuracy and energy) and participation metrics
        self.client_scores = [0 for i in range(config.num_clients)] 
        self.client_participation = [0 for i in range(config.num_clients)]

        # Initialise global model
        self.global_model = InertialTransformer().to(self.device)

        # Initialise personalised models for FedPer
        if config.algorithm == 'fedper':
            self.personalised_dicts = [None for i in range(config.num_clients)]

    def select_clients(self):
        """
        Select clients to participate in global aggregation round

        If self.config.smart_selection is True, use accuracy- and energy-aware 
        client selection. Otherwise, randomly select clients
        """
        all_indices = np.argsort(self.client_scores)
        indices = []
        num_clients = math.ceil((self.config.frac) * self.config.num_clients)

        if self.config.smart_selection == True:
            # Energy and accuracy aware selection
            for i in range(self.config.num_clients):
                for j in range(self.config.devices_per_client):
                    logging.info(f"Client {i} battery (device {j + 1}): {str(self.client_energies[i][j])}")
                logging.info(f"Client {i} loss: {self.client_utils[i]}")
                logging.info(f"Client {i} score: {self.client_scores[i]}\n")
            
            # Select clients with highest score
            logging.info("Select clients with highest scores")
            indices.extend(all_indices[-num_clients:])

        else:
            # Randomly select clients
            logging.info("Randomly select clients")
            random_indices = np.random.choice(all_indices, size=num_clients, replace=False)
            indices.extend(random_indices)

        return indices

    def train_clients(self, selected_clients, global_grads=None):
        """
        Receive local model updates from selected clients

        Args:
            selected_clients: list containing the selected clients for
            global aggregation round
            global_grads: used by FedSGD and FedDANE to prevent drift 
            during local training 
        
        Returns:
            local_models: updated client models using local data
            step_counts: number of steps taken by each client during model
            training
            total_losses: total loss for each client during model training
        """
        local_models = []
        step_counts = []
        total_losses = []
        total_entropy = []
        for idx in selected_clients:
            local_model = copy.deepcopy(self.global_model)

            # Re-load the personalised model for fedper clients
            if self.config.algorithm == 'fedper':
                if self.personalised_dicts[idx] != None:
                    local_model.set_state_dict(self.personalised_dicts[idx])

            if self.config.hetero_wearables:
                selection = self.wearable_selection[idx]
            else:
                selection = None

            client = Client(
                model=local_model,
                train_loader=self.data_loader.get_training_idx(idx),
                config=self.config,
                device=self.device,
                client_id=idx,
                wearable_selection=selection
            )

            trained_model, step_count, utility, entropy = client.train(
                global_model=copy.deepcopy(self.global_model),
                global_grads=global_grads,
            )

            local_models.append(trained_model)
            step_counts.append(step_count)
            total_losses.append(utility)
            total_entropy.append(entropy)

        return local_models, step_counts, total_losses, total_entropy

    def avg_grads(self, local_models, client_weight, client_accs):
        """
        FedAvg
        Average parameters across all local models, weighted by data size and optionally, accuracy
        Update the global model

        Args:
            local_models: models being aggregated
            client_weight: amount of data for each client
            client_accs: local accuracy of each client
        """
        if self.config.accuracy_aware_agg == True:
            # Invert client losses
            accs = torch.tensor(client_accs, dtype=torch.float32)
            acc_score = accs / torch.sum(accs)
            
            # Combine data size and loss to create weight scaler
            data_weights = torch.tensor(client_weight, dtype=torch.float32)
            combined_weights = data_weights * acc_score
            combined_weights = combined_weights / combined_weights.sum()

        else:
            total_weights = sum(client_weight)
            combined_weights = torch.tensor(client_weight, dtype=torch.float32) / total_weights

        global_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            stacked = torch.stack([lm.state_dict()[key].float() for lm in local_models], dim=0)

            weight_shape = [-1] + [1] * (stacked.dim() - 1)
            w = combined_weights.view(weight_shape).to(stacked.device)

            global_dict[key] = (stacked * w).sum(dim=0)

        self.global_model.load_state_dict(global_dict)
    
    def compute_fednova_update(self, local_model, tau_i):
        """
        Compute normalised update for FedNova

        Args:
            local_model: model parameters for client being normalised
            tau_i: number of steps taken by client during local training
        
        Returns:
            d_i: normalised local model parameters
        """
        global_dict = self.global_model.state_dict()
        local_dict = local_model.state_dict()

        d_i = {}

        for k in global_dict.keys():
            d_i[k] = (global_dict[k].float() - local_dict[k].float()) / tau_i

        return d_i

    def avg_grads_fednova(self, local_models, local_steps_list, client_weights, global_lr=1):
        """
        FedNova
        Normalise and scale local updates before aggregation
        Update the global model
        
        Args:
            local_models: models being aggregated 
            local_steps_list: number of steps taken during local model training for
            each client
            client_weights: amount of data for each client
            global_lr: learning rate for fednova updates
        
        """
        global_dict = self.global_model.state_dict()
        total_weights = sum(client_weights)
        client_weights = [client_weights[i] / total_weights for i in range(len(client_weights))]

        # Compute normalised updates
        client_updates = []
        for local_model, tau_i in zip(local_models, local_steps_list):
            d_i = self.compute_fednova_update(local_model, tau_i)
            client_updates.append(d_i)

        # Calculate weighted tau in terms of client weights and steps
        tau_eff = sum(p_i * tau_i for p_i, tau_i in zip(client_weights, local_steps_list))

        # Aggregate local models
        for k in global_dict.keys():

            agg = torch.zeros_like(global_dict[k]).float()

            for p_i, d_i in zip(client_weights, client_updates):
                agg += p_i * d_i[k]

            # FedNova
            global_dict[k] = (global_dict[k].float() - global_lr * tau_eff * agg)

        self.global_model.load_state_dict(global_dict)
    
    def avg_grads_fedper(self, local_models, client_weight):
        """
        FedPer
        Average base layer parameters across all local models, weighted by data size.
        Personalized layer parameters remain local to clients

        Args:
            local_models: models being aggregated
            client_weight: amount of data for each client
        """
        total_weights = sum(client_weight)
        combined_weights = torch.tensor(client_weight, dtype=torch.float32) / total_weights

        global_dict = self.global_model.get_base_state_dict()
        for key in global_dict.keys():
            stacked = torch.stack([lm.get_base_state_dict()[key].float() for lm in local_models], dim=0)

            weight_shape = [-1] + [1] * (stacked.dim() - 1)
            w = combined_weights.view(weight_shape).to(stacked.device)

            global_dict[key] = (stacked * w).sum(dim=0)

        self.global_model.set_state_dict(global_dict)

    def run(self):
        all_metrics = []
        wo_total, wt_total = None, None

        # Report the energy usage for each wearable 
        if self.config.hetero_wearables:
            self.wearable_selection, wo_total, wt_total = calculate_wearable_energy(self.config.num_clients, self.window_counts)
            logging.info(f"Wearable 1 energy usage: {wo_total}")
            logging.info(f"Wearable 2 energy usage: {wt_total}")

        needs_global_grads = self.config.algorithm in ("feddane", "fedsgd")

        # Report the configuration for FL testing
        logging.info(f"Number of rounds: {self.config.num_rounds}")
        logging.info(f"Number of epochs: {self.config.local_epochs}")
        logging.info(f"Fraction of clients: {self.config.frac}")
        logging.info(f"Scaling factor for score: {self.config.f}")
        logging.info(f"Straggler penalty: {self.config.alpha}")
        logging.info(f"Devices per client: {self.config.devices_per_client}")
        logging.info(f"Reduced object sensor data dimensions: {self.config.reduce_dimensions}")
        logging.info(f"One wearable at a time: {self.config.hetero_wearables}")
        logging.info(f"Multiply client weight by acc / total_acc")

        # Initialise variables for FL
        best_acc = -1.0
        best_state = None
        best_round = -1
        min_energy_indices = [-1 for i in range(self.config.num_clients)]

        # Initialise energy usage values
        used_energy = []
        for cli in range(len(self.client_energies)):
            used = 0
            for dev in range(len(self.client_energies[cli])):
                used += self.client_energies[cli][dev].get_used_battery()
            used_energy.append(used)

        # Perform FL
        for r in range(1, self.config.num_rounds + 1):
            client_weight = []

            # Calculate client scores
            for i in range(self.config.num_clients):
                min_energy = np.inf
                total_energy = 0

                # Select the client device that has used the least amount of energy
                for j in range(self.config.devices_per_client):
                    client_energy = self.client_energies[i][j].get_used_battery()
                    # client_energy = self.client_energies[i][j].get_battery()
                    total_energy += client_energy
                    if client_energy < min_energy:
                        min_energy = client_energy
                        min_energy_indices[i] = j
                    
                # Compute the client score
                self.client_scores[i] = calculate_federated_score(self.config.f, self.client_utils[i], np.sum(self.client_utils), total_energy, np.sum(used_energy))
            
            # Select clients for global aggregation round
            selected = self.select_clients()
            for s in selected:
                if s not in self.participated:
                    self.participated.append(s)

            # Receive amount of data available to selected clients for local training
            for idx in selected:
                client_weight.append(len(self.data_loader.get_training_idx(idx)))
            
                # Compute global grads for FedSGD and FedDANE
                global_grads = compute_global_grads(self.global_model, self.data_loader.get_training_idx(idx), self.device) if needs_global_grads else None

            # Evaluate performance of global model (no local training)
            accs, _ = evaluate_local([self.global_model for i in range(self.config.num_clients)], [i for i in range(self.config.num_clients)], self.data_loader, self.config.num_classes, self.device, "before")
            # Allow client to evaluate performance using the global model if it hsa not participated in an aggregation round (EAFL eval)
            for idx, acc in enumerate(self.local_accuracies):
                if (idx not in selected) and (idx not in self.participated):
                    self.local_accuracies[idx] = accs[idx]
            
            # Perform local training and evaluate performance of updated model
            local_models, local_steps, total_losses, total_entropy  = self.train_clients(selected, global_grads)
            accs, metrics = evaluate_local(local_models, selected, self.data_loader, self.config.num_classes, self.device, "after")

            # Calculate a new global model
            if self.config.algorithm == "fednova":
                self.avg_grads_fednova(local_models, local_steps, client_weight)
            elif self.config.algorithm == "fedper":
                curr_client = 0
                for select in selected:
                    self.personalised_dicts[select] = local_models[curr_client].get_personalised_state_dict()
                    curr_client += 1
                self.avg_grads_fedper(local_models, client_weight)
            else:
                self.avg_grads(local_models, client_weight, accs)

            # Update energy and utilisation values for clients
            curr_client = 0
            for i in selected:
                self.client_participation[i] += 1
                self.local_accuracies[i] = accs[curr_client]
                index = min_energy_indices[i]
                self.client_energies[i][index].update_energy()
                # self.client_utils[i] = calculate_util(client_weight[curr_client], self.client_energies[i][index].get_training_time(), self.config.alpha)
                self.client_utils[i] = calculate_util(total_losses[curr_client], 
                                        self.client_energies[i][index].get_training_time(), self.config.alpha)
                curr_client += 1
            
            # Determine the total energy usage for clients
            curr_energy = []
            used_energy = []
            for cli in range(len(self.client_energies)):
                total = 0
                used = 0
                for dev in range(len(self.client_energies[cli])):
                    total += self.client_energies[cli][dev].get_initial_battery()
                    used += self.client_energies[cli][dev].get_used_battery()
                
                used_energy.append(used)
                curr_energy.append((1 - used/total) * 100)
            
            logging.info(f"Used energy: {used_energy}")

            # Add values to metrics for plotting
            metrics["local_accuracy"] = copy.deepcopy(self.local_accuracies)
            metrics["energy"] = curr_energy
            metrics["used_energy"] = used_energy
            metrics["participation"] = copy.deepcopy(self.client_participation)
            all_metrics.append(metrics)

            # Determine most accurate aggregation round and store model
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                best_round = r
                best_state = {k: v.cpu() for k, v in self.global_model.state_dict().items()}

            # Log the average local accurcay
            logging.info(f"Local Accuracies: {np.mean(self.local_accuracies):.2f}")
        
        # Log the best accuracy and round
        logging.info(f"Best accuracy: {best_acc:.2f}% at round {best_round}")
                
        return best_state, best_acc, best_round, all_metrics, self.local_accuracies, wo_total, wt_total
