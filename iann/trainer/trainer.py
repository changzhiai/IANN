import numpy as np
import math, time
import json, os, toml, sys
import argparse
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from iann.data.data import AseDataset, collate_atomsdata
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Default configuration that can be overridden
DEFAULT_CONFIG = {
    "node_size": 128,
    "num_interactions": 3,
    "cutoff": 5.5,
    "val_ratio": 0.1,
    "output_dir": "output",
    "max_steps": 1000000,
    "batch_size": 12,
    "initial_lr": 0.0001,
    "forces_weight": 0.9,
    "log_interval": 2000,
    "normalization": False,
    "atomwise_normalization": False,
    "stop_patience": 200,
    "plateau_scheduler": False,
    "random_seed": 666,
    "split_file": None,
    "load_model": False,
    "max_epochs": None,  # None if setup max_steps, otherwise max_epochs
    "device": None,      # override device, e.g. 'cpu' or 'cuda:1'
    "dist_timeout": 600,     # 30 minutes timeout for distributed operations
    "master_port": 12356,
    "debug": False,
    "optimizer_type": "adam",
    'output_log': 'print_out.log',
    "max_grad_norm": None,    # Gradient clipping norm
}

# Logging filter to inject rank into log records
class RankFilter(logging.Filter):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
    def filter(self, record):
        record.rank = self.rank
        return True

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="IANN arguments", fromfile_prefix_chars="+"
    )     
    parser.add_argument(
        "--cfg",
        type=str,
        help="Path to a toml config file'"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["painn", "nequip", "mace", "equiformer2"],
        help="Type of model to use"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to a dataset file"
    )
    
    return parser.parse_args(arg_list)

def forces_criterion(predicted, target, reduction="mean"):
    diff = predicted - target
    total_squared_norm = torch.linalg.norm(diff, dim=1)  # bs
    if reduction == "mean":
        scalar = torch.mean(total_squared_norm)
    elif reduction == "sum":
        scalar = torch.sum(total_squared_norm)
    else:
        raise ValueError("Reduction must be 'mean' or 'sum'")
    return scalar

def get_normalization(dataset, per_atom=True):
    x_sum = torch.zeros(1, dtype=torch.double)
    x_2 = torch.zeros(1, dtype=torch.double)
    num_objects = 0
    for i, sample in enumerate(dataset):
        if i == 0:
            if per_atom:
                bias = sample.energy / sample.num_atoms
            else:
                bias = sample.energy
        x = sample.energy
        if per_atom:
            x = x / sample.num_atoms
        x -= bias
        x_sum += x
        x_2 += x ** 2.0
        num_objects += 1
    x_mean = x_sum / num_objects
    x_var = x_2 / num_objects - x_mean ** 2.0
    x_mean = x_mean + bias
    default_type = torch.get_default_dtype()

    return x_mean.type(default_type), torch.sqrt(x_var).type(default_type)

class EarlyStopping():
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, best_loss):
        if val_loss - best_loss > self.min_delta:
            self.counter +=1
            if self.counter >= self.patience:  
                self.early_stop = True
                
        return self.early_stop

def split_data(dataset, config):
    # Load or generate splits
    if config["split_file"]:
        with open(config["split_file"], "r") as fp:
            splits = json.load(fp)
    else:
        datalen = len(dataset)
        num_validation = int(math.ceil(datalen * config["val_ratio"]))
        indices = np.random.permutation(len(dataset))
        splits = {
            "train": indices[num_validation:].tolist(),
            "validation": indices[:num_validation].tolist(),
        }
    # Save split file
    if config["debug"]:
        output_dir = config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "DataSplitsLog.json"), "w") as f:
            json.dump(splits, f)

    # Split the dataset
    datasplits = {}
    for key, indices in splits.items():
        datasplits[key] = torch.utils.data.Subset(dataset, indices)
    return datasplits

class Trainer:
    """Trainer class for training interatomic neural network models"""
    
    def __init__(self, model="painn", config=None, distributed=True, rank=None, world_size=None):
        """
        Initialize the trainer with a model type and optional config
        
        Args:
            model (str): Model type ("painn", "nequip", "mace", or "equiformer2")
            config (dict, optional): Configuration overrides
            distributed (bool, optional): Whether to use distributed training
            rank (int, optional): Rank of the current process
            world_size (int, optional): Total number of processes
        """
        # Initialize configuration with defaults
        self.config = DEFAULT_CONFIG.copy()
        
        # Update with user config if provided
        if config:
            self.config.update(config)
            
        # Set model type
        self.model_type = model.lower()
        
        # Auto-detect SLURM to avoid spawning when SLURM is already managing processes
        self._under_slurm = 'SLURM_JOB_ID' in os.environ
        # Track whether rank was provided explicitly
        self.distributed = distributed
        self._explicit_rank = rank is not None
        # Auto-detect SLURM rank/world_size if not provided
        if self.distributed:
            # Rank
            if self._explicit_rank:
                self.rank = rank
            else:
                self.rank = int(os.environ.get('SLURM_PROCID', 0))
            # World size
            if world_size is not None:
                self.world_size = world_size
            else:
                self.world_size = int(os.environ.get('SLURM_NTASKS', 1))
        else:
            self.rank = 0
            self.world_size = 1
        # Initialize device from config override (e.g., 'cpu' or 'cuda:0'), if provided
        self.device = torch.device(self.config["device"]) if self.config.get("device") else None
        self.current_node = None
        
        # Initialize other attributes
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stop = None
        self.dataset = None
        self.datasplits = None
        self.train_loader = None
        self.val_loader = None
        self.train_sampler = None
        self.val_sampler = None
        self.target_mean = None
        self.target_stddev = None

        self.optimizer_type = self.config["optimizer_type"]
        
        # Create output directory
        os.makedirs(self.config["output_dir"], exist_ok=True)
        
        # Logging is configured here, now that rank is set
        self._setup_logging()
        # Set random seed
        setup_seed(self.config["random_seed"])
    
    def _setup_distributed(self):
        """Initialize distributed training environment"""
        if 'SLURM_JOB_NODELIST' in os.environ:
            # Get the first node name from the SLURM node list
            node_list = os.environ['SLURM_JOB_NODELIST']
            # Handle different node list formats
            if '[' in node_list and ']' in node_list:
                base_name = node_list.split('[')[0]
                node_range = node_list.split('[')[1].split(']')[0]
                # Handle range format like "034-035"
                if '-' in node_range:
                    start, end = node_range.split('-')
                    # Preserve leading zeros by using the width of the original strings
                    width = len(start)
                    start_num = int(start)
                    end_num = int(end)
                    node_numbers = [f"{i:0{width}d}" for i in range(start_num, end_num + 1)]
                elif ',' in node_range:
                    node_numbers = node_range.split(',')
                else:
                    node_numbers = [node_range]
                node_numbers = sorted(node_numbers)
                master_addr = f"{base_name}{node_numbers[0]}"  # Use first node as master
                node_index = self.rank % len(node_numbers)  # Use modulo to wrap around if rank > num_nodes
                current_node = f"{base_name}{node_numbers[node_index]}"
            else:
                master_addr = node_list.split(',')[0]
                current_node = master_addr
        else:
            master_addr = os.environ.get('MASTER_ADDR', 'localhost')
            current_node = master_addr
        # Use fixed port for simplicity
        master_port = self.config["master_port"]
        self.master_addr = master_addr
        self.master_port = master_port
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        if self.rank == 0:
            logging.info(f"PyTorch version: {torch.__version__}") 
            logging.info(f"Node List: {os.environ.get('SLURM_JOB_NODELIST', 'N/A')}")
            if torch.cuda.is_available():
                logging.info(f"World Size (number of GPUs): {self.world_size}")
            else:
                logging.info(f"World Size (number of CPUs): {self.world_size}")
            logging.info(f"Master Address: {master_addr}")
            logging.info(f"Master Port: {master_port}")

        time.sleep(self.rank * 0.1 + 0.1) 

        # Set device and self.device
        if torch.cuda.is_available() and self.device.type == 'cuda':
            if 'SLURM_LOCALID' in os.environ:
                local_rank = int(os.environ['SLURM_LOCALID'])
            else:
                local_rank = self.rank % torch.cuda.device_count()
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f"cuda:{local_rank}")
            logging.info(f"Process {self.rank} using device {self.device} on {current_node}. GPU architecture: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            logging.info(f"Process {self.rank} using device {self.device} on {current_node}, CPU architecture: {info['brand_raw']}")
        
        # Choose backend based on device: NCCL for GPUs, Gloo for CPU
        backend = "nccl" if self.device is not None and self.device.type.startswith("cuda") else "gloo"
        
        dist.init_process_group(
            backend,
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=self.config["dist_timeout"])
        )
            
        # Log NCCL configuration for debugging
        if self.rank == 0 and backend == "nccl" and self.config["debug"]:
            logging.info("NCCL Configuration:")
            logging.info(f"  NCCL_TIMEOUT: {os.environ.get('NCCL_TIMEOUT', 'Not set')}")
            logging.info(f"  NCCL_DEBUG: {os.environ.get('NCCL_DEBUG', 'Not set')}")
            logging.info(f"  NCCL_BLOCKING_WAIT: {os.environ.get('NCCL_BLOCKING_WAIT', 'Not set')}")
            logging.info(f"  NCCL_IB_DISABLE: {os.environ.get('NCCL_IB_DISABLE', 'Not set')}")
            logging.info(f"  NCCL_P2P_DISABLE: {os.environ.get('NCCL_P2P_DISABLE', 'Not set')}")

        # Wait for all ranks to finish logging
        time.sleep((self.world_size - self.rank) * 0.1 + 0.1)  # Each rank waits for others
        self.current_node = current_node
        return current_node
    
    def _cleanup_distributed(self):
        """Clean up distributed environment"""
        if self.distributed:
            dist.destroy_process_group()
            sys.exit(0)  # Clean and Pythonic exit
    
    def _setup_data(self, dataset_path):
        """Setup dataset and dataloaders"""
        if self.rank == 0:
            logging.info(f"Loading data from {os.path.abspath(dataset_path)}")
        
        # Load dataset
        self.dataset = AseDataset(
            ase_db=dataset_path,
            cutoff=self.config["cutoff"],
            compute_forces=bool(self.config["forces_weight"]),
        )
        
        # Split data
        self.datasplits = split_data(self.dataset, self.config)
        
        if self.distributed:
            # Setup distributed samplers
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.datasplits["train"],
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(
                self.datasplits["validation"],
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
        else:
            # Use regular samplers
            self.train_sampler = torch.utils.data.RandomSampler(self.datasplits["train"])
            self.val_sampler = torch.utils.data.SequentialSampler(self.datasplits["validation"])
        
        # Setup dataloaders
        self.train_loader = torch.utils.data.DataLoader(
            self.datasplits["train"],
            self.config["batch_size"],
            sampler=self.train_sampler,
            collate_fn=collate_atomsdata,
            num_workers=0,
            pin_memory=True,
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            self.datasplits["validation"], 
            self.config["batch_size"], 
            sampler=self.val_sampler,
            collate_fn=collate_atomsdata,
            num_workers=0,
            pin_memory=True,
        )
        
        if self.rank == 0:
            logging.info('Dataset size: {}, training set size: {}, validation set size: {}'.format(
                len(self.dataset),
                len(self.datasplits["train"]),
                len(self.datasplits["validation"]),
            ))
        
        # Compute normalization if needed
        if self.config["normalization"]:
            if self.rank == 0:
                logging.info("Computing energy mean and variance of the dataset")
            self.target_mean, self.target_stddev = get_normalization(
                self.datasplits["train"], 
                per_atom=self.config["atomwise_normalization"],
            )
            if self.rank == 0:
                logging.debug(f"Mean of energy: {self.target_mean.item():.4f}, standard deviation of energy: {self.target_stddev.item():.4f}")
        else:
            self.target_mean = torch.tensor([0.0])
            self.target_stddev = torch.tensor([1.0])
        
    
    def _create_model(self):
        """Create model based on model_type"""
        # Common model parameters
        common_params = {
            "cutoff": self.config["cutoff"],
            "compute_forces": bool(self.config["forces_weight"]),
        }
        # update all params in common_params
        common_params.update(self.config) # To do: update all params in log
        if self.config["debug"]:
            logging.info(f"All parameters: {common_params}")
        common_params.pop("num_interactions")
        common_params.pop("node_size")
        common_params.pop("normalization")
        common_params.pop("atomwise_normalization")
        common_params.pop("device")
        
        # To do: set the default value for each model via self.config.get("xxx", x)
        if self.model_type == "painn":
            from iann.models.painn import PaiNN
            model = PaiNN(
                num_interactions=self.config["num_interactions"], 
                hidden_state_size=self.config["node_size"],
                normalization=self.config["normalization"],
                target_mean=self.target_mean.tolist() if self.config["normalization"] else [0.0],
                target_stddev=self.target_stddev.tolist() if self.config["normalization"] else [1.0],
                atomwise_normalization=self.config["atomwise_normalization"],
                **common_params
            )
        elif self.model_type == "nequip":
            try:
                from iann.models.nequip import NequIP
            except ImportError:
                raise ImportError("NequIP is not available")
            
            model = NequIP(
                num_interactions=self.config["num_interactions"],
                num_features=self.config["node_size"],
                lmax=self.config.get("lmax", 2),  # Default lmax
                normalization=self.config["normalization"],
                target_mean=self.target_mean.tolist() if self.config["normalization"] else [0.0],
                target_stddev=self.target_stddev.tolist() if self.config["normalization"] else [1.0],
                atomwise_normalization=self.config["atomwise_normalization"],
                **common_params
            )
        elif self.model_type == "mace":
            try:
                from iann.models.mace import MACE
            except ImportError:
                raise ImportError("MACE is not available")
            model = MACE(
                num_interactions=self.config["num_interactions"],
                num_features=self.config["node_size"],
                correlation=self.config.get("correlation", 3),  # Default correlation
                species=self.config.get("species", None),  # Auto-determine from dataset
                normalization=self.config["normalization"],
                target_mean=self.target_mean.tolist() if self.config["normalization"] else [0.0],
                target_stddev=self.target_stddev.tolist() if self.config["normalization"] else [1.0],
                atomwise_normalization=self.config["atomwise_normalization"],
                **common_params
            )
        elif self.model_type == "equiformerv2":
            try:
                from iann.models.equiformerV2 import EquiformerV2
            except ImportError:
                raise ImportError("EquiformerV2 is not available")
            model = EquiformerV2(
                num_interactions=self.config["num_interactions"],
                num_features=self.config["node_size"],
                device=self.device,
                normalization=self.config["normalization"],
                target_mean=self.target_mean.tolist() if self.config["normalization"] else [0.0],
                target_stddev=self.target_stddev.tolist() if self.config["normalization"] else [1.0],
                atomwise_normalization=self.config["atomwise_normalization"],
                **common_params
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model
    
    def _setup_model(self):
        """Setup model, optimizer, and scheduler"""
        # Create model
        self.model = self._create_model()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Wrap with DDP if distributed
        if self.distributed:
            if self.device.type == 'cpu':
                self.model = DDP(self.model)
            else:
                if 'SLURM_LOCALID' in os.environ:
                    local_rank = int(os.environ['SLURM_LOCALID'])
                    self.model = DDP(
                        self.model, 
                        device_ids=[local_rank],
                        gradient_as_bucket_view=True,  # Memory efficiency
                        broadcast_buffers=False,       # Broadcast buffers may cause stuck for MACE or NequiIP
                        static_graph=False,            # Dynamic graph is False by default
                        find_unused_parameters=True, 
                    )
                    # self.model = DDP(self.model, device_ids=[local_rank])
                else:
                    self.model = DDP(self.model, device_ids=[self.rank])
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.rank == 0:
            logging.info(f"Total trainable parameters: {total_params}")
        total_memory = sum(p.element_size() * p.numel() for p in self.model.parameters())
        if self.rank == 0:
            logging.info(f"Total memory: {total_memory / 1024**2:.2f} MB")
        
        # Setup optimizer
        if self.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["initial_lr"])
        elif self.optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["initial_lr"], weight_decay=self.config.get("weight_decay", 1e-2))
        elif self.optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config["initial_lr"], momentum=self.config.get("momentum", 0.9))
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
        self.criterion = torch.nn.MSELoss()
        
        # Add gradient clipping for stability
        if self.config.get("max_grad_norm") is not None:
            self.max_grad_norm = self.config.get("max_grad_norm")
        
        # Setup scheduler
        if self.config["plateau_scheduler"]:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 'min', factor=0.5, patience=10
            )
        else:
            scheduler_fn = lambda step: 0.96 ** (step / 100000)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, scheduler_fn)
        
        # Setup early stopping
        self.early_stop = EarlyStopping(patience=self.config["stop_patience"])
        
        # Initialize steps
        self.init_steps = 0

        # Load model if needed
        if self.config["load_model"]:
            self._load_model()
        
    
    def _load_model(self):
        """Load model from checkpoint"""
        best_model = os.path.join(self.config["output_dir"], 'best_model.pth')
        if os.path.exists(best_model):
            logging.info(f"Loading model from {best_model}")
            if self.distributed:
                if torch.cuda.is_available() and self.device.type == 'cuda':
                    if 'SLURM_LOCALID' in os.environ:
                        local_rank = int(os.environ['SLURM_LOCALID'])
                    else:
                        local_rank = self.rank % torch.cuda.device_count()
                    map_location = torch.device(f"cuda:{local_rank}")
                else: # CPU
                    local_rank = self.rank
                    map_location = torch.device("cpu")
                state_dict = torch.load(best_model, map_location=map_location)
            else:
                state_dict = torch.load(best_model, map_location=self.device)
                
            if self.distributed:
                self.model.module.load_state_dict(state_dict["model"])
            else:
                self.model.load_state_dict(state_dict["model"])

            if state_dict["step"] > 0:
                self.init_steps = state_dict["step"]
                
            self.scheduler.load_state_dict(state_dict["scheduler"])
        else:
            self.config["load_model"] = False
    
    def _save_model(self, filename, total_steps, best_val_loss):
        """Save model checkpoint"""
        model_state = self.model.module.state_dict() if self.distributed else self.model.state_dict()
        
        torch.save(
            {
                "model_type": self.model_type,
                "model": model_state,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": total_steps,
                "best_val_loss": best_val_loss,
                "node_size": self.config["node_size"],
                "num_layer": self.config["num_interactions"],
                "cutoff": self.config["cutoff"],
                "compute_forces": self.config["forces_weight"],
            },
            os.path.join(self.config["output_dir"], filename),
        )
    
    def eval_model(self):
        """Evaluate model on validation set"""
        model = self.model
        was_training = model.training
        
        # Set model to evaluation mode
        model.eval()
        
        energy_running_ae = 0.0
        energy_running_se = 0.0
        forces_running_c_ae = 0.0
        forces_running_c_se = 0.0
        running_loss = 0.0
        count = 0
        forces_count = 0
        
        for batch in self.val_loader:
            device_batch = batch.to(self.device)
            out = model(device_batch)
            count += device_batch.energy.shape[0]
            energy_loss = self.criterion(out.energy, device_batch.energy).detach().cpu().numpy()

            if bool(self.config["forces_weight"]):
                forces_count += device_batch.forces.shape[0]
                forces_loss = forces_criterion(out.forces, device_batch.forces).detach().cpu().numpy()
            else:
                forces_loss = 0.0

            # use mean square loss here
            total_loss = self.config["forces_weight"] * forces_loss + (1 - self.config["forces_weight"]) * energy_loss
            running_loss += total_loss * device_batch.energy.shape[0]
            
            # energy errors
            energy_targets = device_batch.energy.detach().cpu().numpy()
            energy_outputs = out.energy.detach().cpu().numpy()
            energy_running_ae += np.sum(np.abs(energy_targets - energy_outputs), axis=0)
            energy_running_se += np.sum(np.square(energy_targets - energy_outputs), axis=0)

            # force errors
            if bool(self.config["forces_weight"]):
                forces_targets = device_batch.forces.detach().cpu().numpy()
                forces_outputs = out.forces.detach().cpu().numpy()
                forces_diff = forces_targets - forces_outputs
                forces_running_c_ae += np.sum(np.abs(forces_diff))
                forces_running_c_se += np.sum(np.square(forces_diff))
            else:
                forces_running_c_ae = 0.0
                forces_running_c_se = 0.0
        
        # Restore the model's original training state
        if was_training:
            model.train()

        energy_mae = energy_running_ae / count
        energy_rmse = np.sqrt(energy_running_se / count)

        if bool(self.config["forces_weight"]):
            forces_mae = forces_running_c_ae / (forces_count * 3)
            forces_rmse = np.sqrt(forces_running_c_se / (forces_count * 3))
        else:
            forces_mae = 0
            forces_rmse = 0

        total_loss = running_loss / count

        evaluation = {
            "energy_mae": energy_mae,
            "energy_rmse": energy_rmse,
            "forces_mae": forces_mae,
            "forces_rmse": forces_rmse,
            "sqrt(total_loss)": np.sqrt(total_loss),
        }

        return evaluation
    
    def _setup_logging(self):
        """Setup logging with dynamic rank injection"""
        # Clear existing handlers
        root = logging.getLogger()
        for h in root.handlers[:]:
            try:
                h.flush()
            except Exception:
                pass
            root.removeHandler(h)
        root.setLevel(logging.DEBUG)
        
        # Format: include injected %(rank)s
        fmt = "%(asctime)s [RANK%(rank)s] [%(levelname)-5.5s]  %(message)s"
        formatter = logging.Formatter(fmt)
        
        # Create and configure file handler
        log_file = os.path.join(self.config["output_dir"], self.config["output_log"])
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RankFilter(self.rank))
        
        # Create and configure stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(RankFilter(self.rank))
        
        # Attach handlers
        root.addHandler(file_handler)
        root.addHandler(stream_handler)
    
    def train(self, dataset_path=None):
        """
        Train the model on a dataset
        
        Args:
            dataset_path (str): Path to the dataset file
        """
        # Configure logging first, using the detected or provided rank
        self._setup_logging()
        
        # Determine dataset path
        if dataset_path is None:
            dataset_path = self.config.get("dataset")
            if not dataset_path:
                raise ValueError("Dataset path must be provided via train() argument or in config['dataset']")
        
        # If using local multi-GPU (not under SLURM) and rank not explicitly set, spawn worker processes
        if (self.distributed and self.world_size > 1 and not self._explicit_rank 
            and self.rank == 0 and not self._under_slurm):
            mp.spawn(
                process_function,
                args=(self.world_size, self.model_type, self.config, dataset_path),
                nprocs=self.world_size,
                join=True
            )
            return  # Return after spawning processes - only the spawned children continue
        
        # Setup distributed training if enabled
        if self.distributed:
            self._setup_distributed()
        else:
            # Single-GPU or CPU mode
            try:
                node_name = __import__('platform').node()
            except:
                node_name = "unknown"
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.device.type == 'cuda' else "cpu")
            logging.info(f"PyTorch version: {torch.__version__}") 
            logging.info(f"Running in single-{'GPU' if torch.cuda.is_available() and self.device.type == 'cuda' else 'CPU'} Node {node_name}")
            if torch.cuda.is_available():
                logging.info(f"Hardware architecture: {torch.cuda.get_device_name()}")
            else:
                import cpuinfo
                info = cpuinfo.get_cpu_info()
                logging.info(f"Hardware architecture: {info['brand_raw']}")
        
        # Setup dataset and dataloaders
        self._setup_data(dataset_path)
        
        # Setup model, optimizer, and scheduler
        self._setup_model()
        
        # Log detailed model configuration and setup
        if self.rank == 0:
            logging.info("---------------- Configuration Settings ----------------")
            # logging.info("Configuration settings:")
            logging.info(f"Model Type (model): {self.model_type}")
            logging.info(f"Node Size (node_size): {self.config['node_size']}")
            logging.info(f"Number of Interactions (num_interactions): {self.config['num_interactions']}")
            logging.info(f"Cutoff Radius (cutoff): {self.config['cutoff']}")
            logging.info(f"Forces Weight (forces_weight): {self.config['forces_weight']}")
            logging.info(f"Normalization (normalization): {self.config['normalization']}")
            logging.info(f"Atomwise Normalization (atomwise_normalization): {self.config['atomwise_normalization']}")
            logging.info(f"Batch Size (batch_size): {self.config['batch_size']}")
            logging.info(f"Initial Learning Rate (initial_lr): {self.config['initial_lr']}")
            logging.info(f"Max Steps (max_steps): {self.config['max_steps']}")
            logging.info(f"Max Epochs (max_epochs): {self.config['max_epochs']}")
            logging.info(f"Early Stopping Patience (stop_patience): {self.config['stop_patience']}")
            logging.info(f"Plateau Scheduler (plateau_scheduler): {self.config['plateau_scheduler']}")
            logging.info(f"Validation Ratio (val_ratio): {self.config['val_ratio']}")
            logging.info(f"Split File (split_file): {self.config['split_file']}")
            if self.config['debug']:
                logging.info(f"Target Mean (target_mean): {self.target_mean.item():4f}")
                logging.info(f"Target Stddev (target_stddev): {self.target_stddev.item():4f}")
            logging.info(f"Distributed Training (distributed): {self.distributed}")
            if self.distributed:
                logging.info(f"Master Port (master_port): {self.config['master_port']}")
                logging.info(f"Distributed Timeout (dist_timeout) (s): {self.config['dist_timeout']}")
        
        # Initialize counters
        local_steps = 0
        total_steps = 0
        running_loss = 0.0
        running_loss_count = 0
        training_time = 0.0
        prev_loss = None
        best_val_loss = np.inf
        
        if self.config["max_epochs"]:
            self.config["max_steps"] = None
            max_epochs = self.config["max_epochs"]
        # Training loop
        else:
            max_epochs = int(self.config["max_steps"])
        
        if self.rank == 0:
            logging.info("---------------------- Training ------------------------")
        for epoch in range(max_epochs):
            if epoch == max_epochs - 1:
                logging.info(f"Reached maximum epochs ({max_epochs}), stopping training")
                if self.distributed:
                    self._cleanup_distributed()
                return
            
            if hasattr(self.train_sampler, "set_epoch"):
                # For distributed sampler, set epoch for shuffling
                self.train_sampler.set_epoch(epoch)
                
            self.model.train()
            for batch_idx, batch in enumerate(self.train_loader):
                train_start_time = time.time()
                
                device_batch = batch.to(self.device)
                
                # Forward pass and backward pass
                self.optimizer.zero_grad()
                out = self.model(device_batch)
                
                # Calculate losses
                energy_loss = self.criterion(out.energy, device_batch.energy)
                if bool(self.config["forces_weight"]):
                    forces_loss = forces_criterion(out.forces, device_batch.forces)
                else:
                    forces_loss = 0.0
                
                # Total loss
                total_loss = self.config["forces_weight"] * forces_loss + (1 - self.config["forces_weight"]) * energy_loss
                total_loss.backward()
                
                # Apply gradient clipping
                if self.config.get("max_grad_norm") is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get("max_grad_norm"))

                self.optimizer.step()

                # Update running loss
                running_loss += total_loss.detach().cpu().numpy() * batch.energy.shape[0]
                running_loss_count += batch.energy.shape[0]
                training_time += time.time() - train_start_time

                if self.distributed:
                    total_steps = local_steps * self.world_size + self.rank + self.init_steps
                else:
                    total_steps = local_steps + self.init_steps

                # Log training progress
                if (total_steps % self.config["log_interval"] == 0) or ((total_steps + 1) == self.config["max_steps"]):
                    
                    eval_start = time.time()
                    
                    train_loss = running_loss / running_loss_count
                    running_loss = 0.0
                    running_loss_count = 0
                    
                    # Evaluate model
                    eval_dict = self.eval_model()
                    
                    eval_formatted = ", ".join(
                        ["{}={:.3f}".format(k, v) for (k, v) in eval_dict.items()]
                    )
                    eval_loss = np.square(eval_dict["sqrt(total_loss)"])
                    smooth_loss = eval_loss if prev_loss is None else 0.9 * eval_loss + 0.1 * prev_loss
                    prev_loss = smooth_loss

                    eval_time = (time.time() - eval_start) / 60
                    if self.rank == 0:
                        log_msg = (
                            f"step={total_steps}, {eval_formatted}, "
                            f"sqrt(train_loss)={math.sqrt(train_loss):.3f}, "
                            f"patience={self.early_stop.counter:3d}, "
                            f"training time={training_time/60:.3f} min, "
                            f"eval time={eval_time:.3f} min"
                        )
                        logging.info(log_msg)

                    training_time = 0
                    
                    if self.config["plateau_scheduler"]:
                        self.scheduler.step(smooth_loss)
                    
                    # Check early stopping and save best model
                    if not self.early_stop(math.sqrt(smooth_loss), best_val_loss):
                        best_val_loss = math.sqrt(smooth_loss)
                        
                        self._save_model("best_model.pth", total_steps, best_val_loss)
                    else:
                        logging.info(f"Early stopping, training complete")
                        
                        if self.distributed:
                            self._cleanup_distributed()
                        return
                
                # Count steps
                local_steps += 1
                
                if not self.config["plateau_scheduler"]:
                    self.scheduler.step()
                
                if bool(self.config["max_steps"]) and total_steps >= self.config["max_steps"]:
                    logging.info(f"Maximum steps {self.config['max_steps']} reached, training complete")
                    
                    self._save_model("exit_model.pth", total_steps, best_val_loss)
                    if self.distributed:
                        self._cleanup_distributed()
                    return
        
        self._save_model("final_model.pth", total_steps, best_val_loss)
        
        # Clean up distributed training
        if self.distributed:
            self._cleanup_distributed()


def main():
    """Command-line entry point"""
    args = get_arguments()
    config = {}
    
    if bool(args.cfg):
        with open(args.cfg, 'r') as f:
            config = toml.load(f)
    
    # Check if we're running under SLURM
    slurm_distributed = 'SLURM_JOB_NUM_NODES' in os.environ or 'SLURM_NTASKS' in os.environ
    
    if slurm_distributed:
        # Get SLURM environment variables for distributed training
        if 'SLURM_NTASKS' in os.environ:
            world_size = int(os.environ['SLURM_NTASKS'])
        else:
            world_size = 1
            
        if 'SLURM_PROCID' in os.environ:
            rank = int(os.environ['SLURM_PROCID'])
        else:
            rank = 0
            
        if 'SLURM_LOCALID' in os.environ:
            local_rank = int(os.environ['SLURM_LOCALID'])
            # Set device based on local rank
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
        
        # Create trainer with distributed settings
        trainer = Trainer(
            model=args.model_type, 
            config=config,
            distributed=True,
            rank=rank,
            world_size=world_size
        )
        
        # Train the model
        trainer.train(args.dataset)
    else:
        # Detect if we have multiple GPUs
        num_gpus = torch.cuda.device_count()
        
        if num_gpus > 1:
            # Multi-GPU training on a single node
            logging.info(f"Starting multi-GPU training with {num_gpus} GPUs")
            mp.spawn(
                process_function,
                args=(num_gpus, args.model_type, config, args.dataset),
                nprocs=num_gpus,
                join=True
            )
        else:
            # Single-GPU or CPU training
            logging.info("Starting single-GPU training")
            trainer = Trainer(model=args.model_type, config=config, distributed=False)
            trainer.train(args.dataset)

def process_function(rank, world_size, model_type, config, dataset_path):
    """Function to be spawned in each process for multi-GPU training"""
    # Set device for this process
    torch.cuda.set_device(rank)
    
    # Initialize environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    trainer = Trainer(model=model_type, config=config, distributed=True, rank=rank, world_size=world_size)
    trainer.train(dataset_path)

if __name__ == "__main__":
    main()

    """
    # Example usage:
    # CPU mode:
    # 1) create a new run.py file and copy the following code into it
    from iann.train.train import Trainer
    trainer = Trainer(
        model="painn",
        config={"device": "cpu"},
        distributed=False
    )
    trainer.train("path/to/data.traj")

    # 2) create a new run.sh file and copy the following code into it
    # Single-CPU mode:
    #SBATCH --nodes=1 --ntasks-per-node=1
    module load pytorch
    srun python run.py
    And then submit the job to SLURM:
    sbatch run.sh

    # Multi-CPU mode:
    #SBATCH --nodes=2 --ntasks-per-node=4
    module load pytorch
    srun python run.py
    And then submit the job to SLURM:
    sbatch run.sh

    # GPU mode:
    # 1) create a new run.py file and copy the following code into it
    from iann.train.train import Trainer
    trainer = Trainer(
        model="painn",
        distributed=False
    )
    trainer.train("path/to/data.traj")
    
    # 2) create a new run.sh file and copy the following code into it
    # Single-GPU mode:
    #SBATCH --nodes=1 --gpus-per-node=1
    module load pytorch
    srun python run.py
    And then submit the job to SLURM:
    sbatch run.sh 

    # Multi-GPU mode:
    #SBATCH --nodes=2 --gpus-per-node=4
    module load pytorch
    srun python run.py
    And then submit the job to SLURM:
    sbatch run.sh 


    # python mode:
    python train.py \
        --model_type painn \
        --dataset path/to/data.traj \
        --cfg path/to/arguments.toml
    
    
    # torchrun mode (no SLURM):
    torchrun --nproc_per_node=4 train.py \
        --model_type painn \
        --dataset path/to/data.traj \
        --cfg path/to/arguments.toml
    # Here, nproc_per_node defines the number of local CPU or GPU workers.
    # Device selection:
    # - By default, Trainer auto-uses GPUs if torch.cuda.is_available().
    # - To force CPU-only torchrun, either:
    #     export CUDA_VISIBLE_DEVICES=""
    #   or add `device = "cpu"` in your arguments.toml or config.
    # - To target a specific GPU, set config `device = "cuda:IDX"` or export
    #     CUDA_VISIBLE_DEVICES=IDX


    # Note:for multi-GPUs/multi-CPUs mode, Trainer.train() will call mp.spawn() to launch `world_size` workers using the process_function.
    # The parallelization parameters are automatically obtained from the SLURM environment variables.

    """