import argparse
import torch
import numpy as np
from config import FLConfig
from data_storage import DataStorage
from server import Server

from logging import basicConfig, getLogger, StreamHandler, DEBUG, WARNING
from plots import plot_results
import os
import sys
from datetime import datetime

# Data organised by participant
imu_dataset_files = []

# Directory
CUR_DIR = ""

# Set up logging 
def config_logger():
    EXEC_TIME = "fl-testing-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = os.path.join(CUR_DIR, f"logs/{EXEC_TIME}")
    os.makedirs(LOG_DIR, exist_ok=True)  # Create log directory

    formatter = "%(levelname)s: %(asctime)s: %(filename)s: %(funcName)s: %(message)s"
    basicConfig(filename=f"{LOG_DIR}/{EXEC_TIME}.log", level=DEBUG, format=formatter)
    mpl_logger = getLogger("matplotlib")  # Suppress matplotlib logging
    mpl_logger.setLevel(WARNING)
    # Handle logging to both logging and stdout.
    getLogger().addHandler(StreamHandler(sys.stdout))

    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    logger.debug(f"{LOG_DIR}/{EXEC_TIME}.log")

def parse_args():
    p = argparse.ArgumentParser(description="Federated Learning CLI")
    p.add_argument("--algo", type=str, default="fedavg",
                   choices=["fedavg", "feddane", "fedprox", "fedsgd", "fednova"])
    return p.parse_args()


def main():
    config_logger()
    rng = np.random.default_rng(seed=42)
    cli = parse_args()

    # Separate train and test data
    train_files = []
    test_files = []

    for participant in imu_dataset_files:
        index = rng.integers(0, len(participant))
        test_files.append(participant.pop(index))
        train_files.append(participant)

    config = FLConfig(algorithm = cli.algo, num_clients = 10)
    device = torch.device("cuda" if (config.use_cuda and torch.cuda.is_available()) else "cpu")

    # Configure data storage
    data_loader = DataStorage(config, train_files, test_files)

    # Initialize and run FL server
    server = Server(config, data_loader, device)
    best_state, best_acc, best_round, all_metrics, local_accuracies, wo_total, wt_total = server.run()

    # Plot server results
    plot_results(all_metrics, local_accuracies, wo_total, wt_total)

if __name__ == "__main__":
    main()
