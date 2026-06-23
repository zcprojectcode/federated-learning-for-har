SEED = 43
import os
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
import argparse
import torch
import numpy as np
from config import FLConfig
from data_storage import DataStorage
from server import Server

from logging import basicConfig, getLogger, StreamHandler, DEBUG, WARNING
from plots import plot_results
import sys
import random
import copy
from datetime import datetime

# Data organised by participant
imu_dataset_files = # Filepath to Pytorch tensors containing data

# Directory
CUR_DIR = # Filepath to current directory 

# Set up logging 
def config_logger():
    EXEC_TIME = "A-SingleWearable-ReferenceEnergy-" + datetime.now().strftime("%Y%m%d-%H%M%S")
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

    return LOG_DIR

def parse_args():
    p = argparse.ArgumentParser(description="FL Command Line Interface")
    p.add_argument("--algo", type=str, default="fedavg",
                   choices=["fedavg", "feddane", "fedprox", "fedsgd", "fednova", "fedper"])
    return p.parse_args()


def main():
    LOG_DIR = config_logger()

    # Set seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rng = np.random.default_rng(seed=SEED)
    fl_cli = parse_args()

    # Separate train and test data
    train_files = []
    test_files = []

    for participant in copy.deepcopy(imu_dataset_files):
        index = rng.integers(0, len(participant))
        test_files.append(participant.pop(index))
        train_files.append(participant)

    config = FLConfig(algorithm = fl_cli.algo, num_clients = 10)
    device = torch.device("cuda" if (config.use_cuda and torch.cuda.is_available()) else "cpu")

    # Load per-client train loaders, central test loader and window counts
    data_loader = DataStorage(config, train_files, test_files)

    # Initialize and run federated server
    server = Server(config, data_loader, device, SEED)
    best_state, best_acc, best_round, all_metrics, local_accuracies, wo_total, wt_total = server.run()

    # Plot server results
    plot_results(all_metrics, local_accuracies, wo_total, wt_total, save_dir=LOG_DIR)

if __name__ == "__main__":
    main()
