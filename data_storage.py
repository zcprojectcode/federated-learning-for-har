import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset, TensorDataset
from pathlib import Path
import numpy as np
import logging

# Convert data to binary
THRESHOLDS = {
    18: 2,
    19: 10,
    20: 5,
    21: 5,
    22: 10,
    23: 8,
    24: 5,
    25: 5,
    26: 2,
    27: 5,
    28: 8,
    29: 5,
}

"""
Store the training and test data for individual clients
"""
class DataStorage:
    def __init__(self, config, train_files, test_files):
        self.train_loaders, self.test_loaders, self.total_windows = self.load_client_data(config, train_files, test_files)
    
    def get_training_data(self):
        return self.train_loaders

    def get_training_idx(self, idx):
        return self.train_loaders[idx]
    
    def get_testing_data(self):
        return self.test_loaders
    
    def get_testing_idx(self, idx):
        return self.test_loaders[idx]
    
    def get_windows(self):
        return self.total_windows

    def load_and_filter(self, file_path):
        """
        Import data from file

        Args:
            file_path: location where data is stored
        
        Returns:
            samples: array of data samples
        """
        checkpoint = torch.load(Path(file_path).expanduser().resolve(), weights_only=False)
        samples = []

        # Combine prepare meal and eat
        for imu, label in checkpoint["samples"]:
            if label == 8:
                label = 3
            elif label > 8:
                label -= 1

            # Use the minimal sensor architecture 
            samples.append((np.nan_to_num(imu[:, np.r_[0:9, 18:27, 99:108, 117:135, 144:162, 180:207, 225:261]], nan=0.0), label))
            # samples.append((np.nan_to_num(imu[:, np.r_[0:9, 18:27, 99, 117, 126, 144, 153, 180, 189, 198, 225, 234, 243, 252]], nan=0.0), label))
            
        return samples
    
    def window_samples(self, samples, window_size, window_shift):
        """
        Split the data into windows

        Args:
            samples: individual data samples
            window_size: number of samples in each window
            window_shift: number of overlapping samples between windows
        
        Returns:
            windows: array containing windows of data
            labels: array containing corresponding labels for each window
            window_counts: total number of windows
        """
        windows, labels, window_counts = [], [], []
        for imu, label in samples:
            window_count = 0
            seq_len = imu.shape[0]
            for start in range(0, seq_len - window_size + 1, window_shift):
                imu_window = imu[start:start + window_size].copy()

                windows.append(imu_window)
                labels.append(int(label))
                window_count += 1
            
            window_counts.append(window_count)

        return np.array(windows), np.array(labels), window_counts
    
    def normalise(self, X):
        """
        Feature-wise Z-score normalisation

        Args:
            X: windows of data
        
        Returns: 
            Z-score normalised windows
        """
        all_steps = X.reshape(-1, X.shape[-1])
        mean = all_steps.mean(axis=0)
        std = all_steps.std(axis=0)
        std[std == 0] = 1
    
        return (X - mean) / std
    
    def make_loader(self, X, y, batch_size, shuffle):
        """
        Convert arrays of data and labels to a data loader

        Args:
            X: windows of data
            y: data labels
        
        Returns:
            Data loader containing dataset and batchsize
        """
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def convert_to_binary(self, X):
        """
        Apply thresholds to object sensor data to convert to binary

        Args:
            X: windows of data
        
        Returns:
            X: windows of data where object-mounted sensor data
            is converted to binary
        """
        for col, threshold in THRESHOLDS.items():
            X[:, col] = np.where(np.abs(X[:, col]) < threshold, 0, 1)
        return X

    def load_client_data(self, config, train_files, test_files):
        """
        Window and normalise the data for model training and testing

        Args:
            train_files: training data
            test_files: testing data
        
        Returns:
            Windowed and normalised training data for each client
            Windowed and normalised testing data for each client
            Total number of data windows for each client
        """
        # Client data loaders
        train_loaders = []
        total_windows = []
        for participant in train_files:
            # Load the samples for all training repetitions
            samples = []
            for file_path in participant:
                samples.extend(self.load_and_filter(file_path))

            # Window and normalise the samples
            X, y, window_counts = self.window_samples(samples, config.window_size, config.window_shift)
            total_windows.append(sum(window_counts))
            X = self.normalise(X)
            if config.reduce_dimensions:
                X = self.convert_to_binary(X)

            # Append to set of train loaders
            train_loaders.append(self.make_loader(X, y, config.batch_size, shuffle=True))
        
        logging.info(f"Number of training loaders: {len(train_loaders)}")
        logging.info(f"Test files: ")

        # Individual test sets for local testing
        test_loaders = []
        for file_path in test_files:
            sample = self.load_and_filter(file_path)

            # Window and normalise the samples
            X, y, _ = self.window_samples(sample, config.window_size, config.window_shift)
            X = self.normalise(X)
            if config.reduce_dimensions:
                X = self.convert_to_binary(X)

            # Append to set of test loaders
            test_loaders.append(self.make_loader(X, y, config.batch_size, shuffle=True))
        
        logging.info(f"Number of individual test loaders: {len(test_loaders)}")

        return train_loaders, test_loaders, total_windows
