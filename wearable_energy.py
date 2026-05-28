import random
import logging

def calculate_wearable_energy(num_clients, client_data_len):
    """
    Compute wearable sensor energy

    Args:
        num_clients: total number of clients in the dataset
        client_data_len: number of windows of data for each client
    
    Returns:
        wearable selection: list containing binary values. 0 indicates
            wearable 1 selected and 1 indicated wearable 2 selected for sensing
        wearable_one_energy: list of the total amount of energy used by each 
            wearable for each client for wearable 1
        wearable_two_energy: list of the total amount of energy used by each
            wearable for each client for wearable 2
    """
    wearable_selection = []
    wearable_one_energy = []
    wearable_two_energy = []
    delta = 5

    # Initialise energy loss per round of sensing and communication
    wearable_one_loss = 108.8 + random.uniform(-delta, delta) # mJ
    wearable_two_loss = 100.4 + random.uniform(-delta, delta) # mJ

    for client in range(num_clients):
        client_energy = []

        # Configure initial sensor energies
        wearable_one_total = random.uniform(0, delta)
        wearable_two_total = random.uniform(0, delta)

        for i in range(client_data_len[client]):
            # Decrement sensor energies 
            if wearable_one_total > wearable_two_total: # Collect data on wearable 2
                client_energy.append(1)
                wearable_two_total += wearable_two_loss
            else:                                       # Collect data on wearable 1
                client_energy.append(0)
                wearable_one_total += wearable_one_loss
        
        wearable_selection.append(client_energy)
        wearable_one_energy.append(wearable_one_total)
        wearable_two_energy.append(wearable_two_total)
        logging.info(f"Wearable selections: {client_energy}")
    
    return wearable_selection, wearable_one_energy, wearable_two_energy