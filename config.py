class FLConfig:
    def __init__(self, algorithm: str = "fedavg", num_clients: int = 30):
        # Federated learning settings
        self.algorithm = algorithm
        self.num_clients = num_clients
        self.num_rounds = 12            # Global aggregation rounds
        self.local_epochs = 4           # Local epochs per client
        self.frac = 0.3                 # Fraction of clients each round
        self.local_lr = 3e-4            # Learning rate for local updates
        self.weight_decay = 1e-4        # Weight decay for training
        self.global_lr = 1e-3           # Global learning rate (not used for FedAvg)
        self.batch_size = 128           # Batch size for training
        self.window_size = 600          # Window size for training 
        self.window_shift = 300         # Window shift for training
        self.num_classes = 16           # Number of classes in the dataset
        self.use_cuda = True            # Use GPU if available
        self.f = 0.25                   # Scaling factor for accuracy and energy awareness
        self.alpha = 1                  # Federated learning straggler penalty
        self.devices_per_client = 1     # Number of devices per client 
        self.smart_selection = True     # If True, use accuracy and energy-aware device selection (random if False)
        self.accuracy_aware_agg = False # If True, include accuracy term in model aggregation (FedAvg)
        self.reduce_dimensions = False  # Reduce the object sensor dimensions to binary
        self.hetero_wearables = False   # If true, select one wearable sensor based on energy

        # Where to save the global model
        self.save_path = f"models/global_{algorithm}.pth"
