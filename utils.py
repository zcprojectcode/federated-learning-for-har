import numpy as np
import torch
import torch.nn as nn

def calculate_federated_score(f, util, total_util, energy, total_energy):
    """ 
    Calculate the score for each client for the federated learning scheme
    to maximise energy and accuracy.

    u(i) = f x (util / total_util) + (1-f) x (1 - energy / total_energy)

    Args:
        f: scaling term (0 - 1)
        util: training loss of the client across each sample
              with a time penalty 
        energy: amount of energy used by a client device
        total_energy: total energy used by all client devices 
    
    Returns:
        Score to determine whether the client
        should be included in the next round of model training
    """
    if (total_energy == 0) and (total_util == 0):
        return 0
    elif total_energy == 0:
        return f * (util / total_util)
    elif total_util == 0:
        return (1 - f) * (1 - energy / total_energy)
    else: 
        return f * (util / total_util) + (1 - f) * (1 - energy / total_energy)
        # return f * util + (1 - f) * energy

def calculate_util(loss, time, alpha):
    """ 
    Calculate the utilisation for each client in terms of loss and
    training time

    Args: 
        loss: client RMS training loss
        time: client training time
    
    Returns: 
        Training loss of the lient across each sample 
        with a time penalty 
    """

    time_scaler = 0

    if time > 45:
        time_scaler = 1
    else:
        time_scaler = (45 / time) ** alpha

    return loss * time_scaler

def compute_global_grads(global_model, train_loader, device):
    """
    Compute gradients for the global model for FedDANE and FedSGD

    Args:
        global_model: current state of the global model downloaded from the
                      server
        train_loader: client training data
        device: training platform
    
    Returns: 
        Change in model parameters (gradients) during model training
    """
    global_model.train()
    global_model.zero_grad()
    criterion = nn.CrossEntropyLoss()
    total_samples = 0

    # Determine gradients 
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits, _ = global_model(x)
        loss = criterion(logits, y) * len(y)
        loss.backward()
        total_samples += len(y)

    # Normalise accumulated gradients by total sample count
    with torch.no_grad():
        for p in global_model.parameters():
            if p.grad is not None:
                p.grad /= total_samples

    return [
        p.grad.clone() if p.grad is not None else torch.zeros_like(p)
        for p in global_model.parameters()
    ]
