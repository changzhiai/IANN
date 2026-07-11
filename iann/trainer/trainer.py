from iann.data import AseDataset, collate_atomsdata
import numpy as np
import math, time
import json, os, toml, sys
import argparse
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)


path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Default configuration that can be overridden
DEFAULT_CONFIG = {
    # parameters for model
    "num_channels": 128, # number of channels in the model
    "num_layers": 3, # number of layers in the model
    "cutoff": 5.5, # cutoff radius
    # parameters for trainer
    "device": None,      # override device, e.g. 'cpu' or 'cuda'
    "val_ratio": 0.1, # validation ratio
    "batch_size": 12, # batch size
    "learning_rate": 0.0001, # initial learning rate
    "forces_weight": 0.9, # weight for forces
    "virial_weight": 0.0, # weight for virial
    "stress_weight": 0.0, # weight for stress
    "loss_per_atom": True, # whether to scale loss by number of atoms
    "load_model": False, # load model from checkpoint
    "max_steps": 1000000, # maximum number of steps
    "max_epochs": None,  # None if setup max_steps, otherwise max_epochs
    "optimizer_type": "adam", # optimizer type: "adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "adamw"
    "max_grad_norm": 10.0,    # gradient clipping norm
    "log_interval": 2000, # log interval
    "stop_patience": 200, # patience for early stopping
    "scheduler_type": "LambdaLR", # scheduler type: "ReduceLROnPlateau", "LambdaLR", "CosineAnnealingLR", "CosineAnnealingWarmRestarts", "StepLR", "MultiStepLR", "ExponentialLR"
    # parameters for data
    "random_seed": 666, # random seed for reproducibility
    "save_split": False, # save split file name
    "load_split": False, # load split file name
    "norm_data": False, # normalize data
    "norm_per_atom": False, # normalize data per atom
    "norm_sample_size": None, # if int, estimate norm stats from a random subsample of this size (None = use all training data)
    "norm_num_workers": 0, # parallel I/O threads for computing norm stats (0 = serial)
    # parameters for DDP (Parallelization)
    "dist_timeout": 600,  # timeout (seconds) for distributed operations
    "master_port": 12356, # port for distributed operations
    # parameters for output
    "output_dir": "output", # output directory
    "output_log": "output.log", # log file
    "output_model": "model.pt", # model file
    "log_input": False, # log input config
    "debug": False, # debug mode
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
        choices=["painn", "nequip", "mace", "equiformer2", "equiformer3", "allegro", "uma", "fastpot", "demo"],
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

def get_norm_data(dataset, per_atom=True, sample_size=None, num_workers=0, seed=0):
    """
    Compute energy mean and standard deviation over `dataset`.

    Parameters
    ----------
    dataset : AseDataset or torch.utils.data.Subset wrapping one
    per_atom : bool
        Normalize each energy by num_atoms before accumulating.
    sample_size : Optional[int]
        If set, estimate stats from a random subsample of this size.
    num_workers : int
        Parallel I/O threads (0 = serial). Threads are used (not processes)
        because ase ``Trajectory`` objects are not reliably picklable and
        frame reads release the GIL during file I/O.
    seed : int
        Seed for the subsampling RNG (only used if ``sample_size`` is set).
    """
    indices = None
    if isinstance(dataset, torch.utils.data.Subset):
        indices = list(dataset.indices)
        dataset = dataset.dataset
    db = getattr(dataset, "db", None)  # ase.io.Trajectory inside AseDataset

    n_total = len(indices) if indices is not None else len(dataset)
    if sample_size is not None and 0 < sample_size < n_total:
        rng = np.random.default_rng(seed)
        sel = rng.choice(n_total, size=sample_size, replace=False)
        indices = (sel.tolist() if indices is None
                   else [indices[int(i)] for i in sel])

    iter_indices = indices if indices is not None else range(n_total)
    n = len(iter_indices)
    if n == 0:
        raise ValueError("get_norm_data: empty dataset")

    energies = np.empty(n, dtype=np.float64)
    num_atoms = np.empty(n, dtype=np.int64)

    def _read(idx):
        a = db[idx]
        return a.get_potential_energy(), a.get_global_number_of_atoms()

    if db is not None and num_workers and num_workers > 0:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for i, (e, na) in enumerate(pool.map(_read, iter_indices)):
                energies[i] = e
                num_atoms[i] = na
    elif db is not None:
        for i, idx in enumerate(iter_indices):
            energies[i], num_atoms[i] = _read(idx)
    else:
        for i, idx in enumerate(iter_indices):
            s = dataset[idx]
            energies[i] = float(s.energy)
            num_atoms[i] = int(s.num_atoms)

    y = energies / num_atoms if per_atom else energies
    mean = float(y.mean())
    std = float(y.std())  # population std, ddof=0

    default_type = torch.get_default_dtype()
    return (torch.tensor([mean], dtype=default_type),
            torch.tensor([std], dtype=default_type))

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

def split_data(dataset, config, rank):
    # Load existed split file
    if config["load_split"]:
        with open(f"{config['load_split']}.json", "r") as fp:
            splits = json.load(fp)
            if rank == 0:
                logging.info(f"Loaded Existed Json Split File (load_split): {config['load_split']}")
    else:
        # Split the dataset into training and validation sets
        datalen = len(dataset)
        num_validation = int(math.ceil(datalen * config["val_ratio"]))
        indices = np.random.permutation(len(dataset))
        splits = {
            "train": indices[num_validation:].tolist(),
            "validation": indices[:num_validation].tolist(),
        }
    # Save split file
    if config["save_split"]:
        output_dir = config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{config['save_split']}.json"), "w") as f:
            json.dump(splits, f)
        if rank == 0:
            logging.info(f"Saved Json Split File (save_split): {config['save_split']}")

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
        
        self.input_config = config
        
        # Set model type
        self.model_type = model.lower()
        if self.model_type not in ["painn", "nequip", "mace", "equiformerv2", "equiformerv3", "allegro", "uma", "fastpot", "demo"]:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
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
        self.data_mean = None
        self.data_stddev = None

        self.optimizer_type = self.config["optimizer_type"]
        self.scheduler_type = self.config["scheduler_type"]

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
            compute_stress=bool(self.config["stress_weight"]),
            compute_virial=bool(self.config["virial_weight"]),
        )
        
        # Split data
        self.datasplits = split_data(self.dataset, self.config, self.rank)
        
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
        
        # Compute data normalization if norm_data or norm_per_atom is True
        default_type = torch.get_default_dtype()
        if self.config["norm_data"] or self.config["norm_per_atom"]:
            if self.rank == 0:
                per_atom = bool(self.config["norm_per_atom"])
                sample_size = self.config.get("norm_sample_size")
                num_workers = int(self.config.get("norm_num_workers", 0) or 0)
                n_train = len(self.datasplits["train"])

                workers_suffix = f" (num_workers={num_workers})" if num_workers else ""
                if sample_size:
                    logging.info(
                        f"Computing energy mean/std on a random subsample of "
                        f"{min(int(sample_size), n_train)}/{n_train} training frames"
                        f"{workers_suffix}"
                    )
                else:
                    logging.info(
                        f"Computing energy mean/std over all training frames"
                        f"{workers_suffix}..."
                    )
                t0 = time.time()
                mean_t, std_t = get_norm_data(
                    self.datasplits["train"],
                    per_atom=per_atom,
                    sample_size=sample_size,
                    num_workers=num_workers,
                    seed=self.config["random_seed"],
                )
                self.data_mean = mean_t.to(default_type)
                self.data_stddev = std_t.to(default_type)
                logging.info(
                    f"Mean of energy: {self.data_mean.item():.4f}, "
                    f"standard deviation: {self.data_stddev.item():.4f} "
                    f"(computed in {time.time() - t0:.2f}s)"
                )
            else:
                # Placeholders; DDP will overwrite these on the model.
                self.data_mean = torch.tensor([0.0], dtype=default_type)
                self.data_stddev = torch.tensor([1.0], dtype=default_type)
        else:
            self.data_mean = torch.tensor([0.0])
            self.data_stddev = torch.tensor([1.0])
        
        if self.rank == 0:
            if bool(self.config['forces_weight']):
                logging.info("Compute forces: True")
            else:
                logging.info("Compute forces: False")
                
            if bool(self.config['stress_weight']):
                logging.info("Compute stress: True")
            if bool(self.config['virial_weight']):
                logging.info("Compute virial: True")

    def _create_model(self):
        """Create model based on model_type"""
        # Other model parameters
        model_params = {
            "compute_forces": bool(self.config["forces_weight"]),
            "compute_virial": bool(self.config["virial_weight"]),
            "compute_stress": bool(self.config["stress_weight"]),
        }
        # update all params in model_params
        model_params.update(self.config)
        model_params.pop("num_layers")
        model_params.pop("num_channels")
        model_params.pop("norm_data")
        model_params.pop("norm_per_atom")
        model_params.pop("device")
        
        # Set model
        if self.model_type == "painn":
            try:
                from iann.models.painn import PaiNN
            except ImportError:
                raise ImportError("PaiNN is not available")
            model = PaiNN(
                num_layers=self.config["num_layers"], 
                num_channels=self.config["num_channels"],
                norm_data=self.config["norm_data"],
                data_mean=self.data_mean.tolist() if self.config["norm_data"] else [0.0],
                data_stddev=self.data_stddev.tolist() if self.config["norm_data"] else [1.0],
                norm_per_atom=self.config["norm_per_atom"],
                **model_params
            )
        elif self.model_type == "nequip":
            try:
                from iann.models.nequip import NequIP
            except ImportError:
                raise ImportError("NequIP is not available")
            
            model = NequIP(
                num_layers=self.config["num_layers"],
                num_channels=self.config["num_channels"],
                norm_data=self.config["norm_data"],
                data_mean=self.data_mean.tolist() if self.config["norm_data"] else [0.0],
                data_stddev=self.data_stddev.tolist() if self.config["norm_data"] else [1.0],
                norm_per_atom=self.config["norm_per_atom"],
                **model_params
            )
        elif self.model_type == "mace":
            try:
                from iann.models.mace import MACE
            except ImportError:
                raise ImportError("MACE is not available")
            model = MACE(
                num_layers=self.config["num_layers"],
                num_channels=self.config["num_channels"],
                norm_data=self.config["norm_data"],
                data_mean=self.data_mean.tolist() if self.config["norm_data"] else [0.0],
                data_stddev=self.data_stddev.tolist() if self.config["norm_data"] else [1.0],
                norm_per_atom=self.config["norm_per_atom"],
                **model_params
            )
        elif self.model_type == "equiformerv2":
            try:
                from iann.models.equiformerV2 import EquiformerV2
            except ImportError:
                raise ImportError("EquiformerV2 is not available")
            model = EquiformerV2(
                num_layers=self.config["num_layers"],
                num_channels=self.config["num_channels"],
                device=self.device,
                norm_data=self.config["norm_data"],
                data_mean=self.data_mean.tolist() if self.config["norm_data"] else [0.0],
                data_stddev=self.data_stddev.tolist() if self.config["norm_data"] else [1.0],
                norm_per_atom=self.config["norm_per_atom"],
                **model_params
            )
        elif self.model_type == "equiformerv3":
            try:
                from iann.models.equiformerV3 import EquiformerV3
            except ImportError:
                raise ImportError("EquiformerV3 is not available")
            model = EquiformerV3(
                num_layers=self.config["num_layers"],
                num_channels=self.config["num_channels"],
                device=self.device,
                norm_data=self.config["norm_data"],
                data_mean=self.data_mean.tolist() if self.config["norm_data"] else [0.0],
                data_stddev=self.data_stddev.tolist() if self.config["norm_data"] else [1.0],
                norm_per_atom=self.config["norm_per_atom"],
                **model_params
            )
        elif self.model_type == "allegro":
            try:
                from iann.models.allegro import Allegro
            except ImportError:
                raise ImportError("Allegro is not available")
            model = Allegro(
                num_layers=self.config["num_layers"],
                num_channels=self.config["num_channels"],
                norm_data=self.config["norm_data"],
                data_mean=self.data_mean.tolist() if self.config["norm_data"] else [0.0],
                data_stddev=self.data_stddev.tolist() if self.config["norm_data"] else [1.0],
                norm_per_atom=self.config["norm_per_atom"],
                **model_params
            )
        elif self.model_type == "uma":
            try:
                from iann.models.uma import UMA
            except ImportError:
                raise ImportError("UMA is not available")
            model = UMA(
                num_layers=self.config["num_layers"],
                num_channels=self.config["num_channels"],
                device=self.device,
                norm_data=self.config["norm_data"],
                data_mean=self.data_mean.tolist() if self.config["norm_data"] else [0.0],
                data_stddev=self.data_stddev.tolist() if self.config["norm_data"] else [1.0],
                norm_per_atom=self.config["norm_per_atom"],
                **model_params
            )
        elif self.model_type == "fastpot":
            try:
                from iann.models.fastpot import FastPot
            except ImportError:
                raise ImportError("FastPot is not available")
            model = FastPot(
                num_layers=self.config["num_layers"],
                num_channels=self.config["num_channels"],
                norm_data=self.config["norm_data"],
                data_mean=self.data_mean.tolist() if self.config["norm_data"] else [0.0],
                data_stddev=self.data_stddev.tolist() if self.config["norm_data"] else [1.0],
                norm_per_atom=self.config["norm_per_atom"],
                **model_params
            )
        elif self.model_type == "demo":
            try:
                from iann.models.demo import Demo
            except ImportError:
                raise ImportError("Demo is not available")
            model = Demo(
                num_layers=self.config["num_layers"],
                num_channels=self.config["num_channels"],
                norm_data=self.config["norm_data"],
                data_mean=self.data_mean.tolist() if self.config["norm_data"] else [0.0],
                data_stddev=self.data_stddev.tolist() if self.config["norm_data"] else [1.0],
                norm_per_atom=self.config["norm_per_atom"],
                **model_params
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model
    
    def _compute_avg_num_neighbors(self, sample_size=1000):
        """Compute average number of neighbors from training data."""
        dataset = self.datasplits["train"]
        n_samples = min(sample_size, len(dataset))
        rng = np.random.default_rng(self.config["random_seed"])
        indices = rng.choice(len(dataset), size=n_samples, replace=False)

        total_edges = 0
        total_atoms = 0
        for idx in indices:
            sample = dataset[int(idx)]
            total_edges += sample.num_edges.item()
            total_atoms += sample.num_atoms.item()

        avg_num_neighbors = total_edges / total_atoms if total_atoms > 0 else 1.0
        return avg_num_neighbors

    def _setup_model(self):
        """Setup model, optimizer, and scheduler"""
        # Compute avg_num_neighbors for models that need it
        if self.model_type in ("mace", "nequip", "allegro"):
            avg_num_neighbors = self._compute_avg_num_neighbors()
            self.config["avg_num_neighbors"] = avg_num_neighbors
            if self.rank == 0:
                logging.info(f"Average number of neighbors: {avg_num_neighbors:.1f}")

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
                    if self.model_type in ("equiformerv2", "equiformerv3") and bool(self.config["forces_weight"]):
                        find_unused_parameters = True
                    else:
                        find_unused_parameters = False
                    self.model = DDP(
                        self.model, 
                        device_ids=[local_rank],
                        gradient_as_bucket_view=True,  # Memory efficiency
                        broadcast_buffers=False,       # Broadcast buffers may cause stuck for MACE or NequiIP
                        static_graph=False,            # Dynamic graph is False by default
                        find_unused_parameters=find_unused_parameters, 
                    )
                else:
                    self.model = DDP(self.model, device_ids=[self.rank])
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.rank == 0:
            logging.info(f"Total trainable parameters: {total_params}")
        total_memory = sum(p.element_size() * p.numel() for p in self.model.parameters())
        if self.rank == 0:
            logging.info(f"Total memory of the model: {total_memory / 1024**2:.2f} MB")
        
        # Setup optimizer
        if self.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        elif self.optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config.get("weight_decay", 1e-2))
        elif self.optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config["learning_rate"], momentum=self.config.get("momentum", 0.9))
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
        self.criterion = torch.nn.MSELoss()
        
        # Add gradient clipping for stability
        if self.config.get("max_grad_norm") is not None:
            self.max_grad_norm = self.config.get("max_grad_norm")
        
        # Setup scheduler
        if self.scheduler_type=="ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 'min',
                factor=self.config.get("scheduler_factor", 0.5),
                patience=self.config.get("scheduler_patience", 10),
                min_lr=self.config.get("scheduler_min_lr", 0.0))
        elif self.scheduler_type=="LambdaLR":
            scheduler_fn = lambda step: 0.96 ** (step / 100000)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, scheduler_fn)
        elif self.scheduler_type=="CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config["max_steps"])
        elif self.scheduler_type=="CosineAnnealingWarmRestarts":
            T_0 = self.config.get("scheduler_T0", 100000)
            T_mult = self.config.get("scheduler_T_mult", 2)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=T_0, T_mult=T_mult)
        elif self.scheduler_type=="StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        elif self.scheduler_type=="MultiStepLR":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 20, 30], gamma=0.5)
        elif self.scheduler_type=="ExponentialLR":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
        
        # Setup early stopping
        self.early_stop = EarlyStopping(patience=self.config["stop_patience"])
        
        # Initialize steps
        self.init_steps = 0

        # Load model if needed
        if self.config["load_model"]:
            self._load_model()
        
    
    def _load_model(self):
        """Load model from checkpoint"""
        if self.config['load_model'].endswith('.pt'):
            best_model = self.config['load_model']
        else:
            best_model = os.path.join(self.config["output_dir"], self.config["output_model"])
        
        if os.path.exists(best_model):
            if self.rank == 0:
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

            # Override LR with config value (allows changing LR on restart)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config["learning_rate"]
                param_group['initial_lr'] = self.config["learning_rate"]
            if self.scheduler_type == "ReduceLROnPlateau":
                # scheduler state_dict carries factor/patience/min_lrs, re-apply configured values
                self.scheduler.factor = self.config.get("scheduler_factor", 0.5)
                self.scheduler.patience = self.config.get("scheduler_patience", 10)
                self.scheduler.min_lrs = [self.config.get("scheduler_min_lr", 0.0)] * len(self.optimizer.param_groups)
            else:
                self.scheduler.base_lrs = [self.config["learning_rate"]] * len(self.scheduler.base_lrs)
        else:
            if self.rank == 0:
                logging.info(f"No model found at {best_model}")
            self.config["load_model"] = False
    
    def _save_model(self, filename, total_steps, best_val_loss):
        """Save model checkpoint"""
        # Check for NaN values and exit if found
        if math.isnan(best_val_loss):
            if self.rank == 0:
                logging.error("NaN values detected in best_val_loss. Exiting training.")
            if self.distributed:
                self._cleanup_distributed()
            sys.exit(1)
        
        model_state = self.model.module.state_dict() if self.distributed else self.model.state_dict()
        
        torch.save(
            {
                "model_type": self.model_type,
                "model": model_state,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": total_steps,
                "best_val_loss": best_val_loss,
                "num_channels": self.config["num_channels"],
                "num_layers": self.config["num_layers"],
                "cutoff": self.config["cutoff"],
                "lmax": self.config.get("lmax"),
                "parity": self.config.get("parity"),
                "use_cue": self.config.get("use_cue"),
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
        virial_running_ae = 0.0
        virial_running_se = 0.0
        stress_running_ae = 0.0
        stress_running_se = 0.0
        running_loss = 0.0
        count = 0
        forces_count = 0
        virial_count = 0
        stress_count = 0
        
        for batch in self.val_loader:
            device_batch = batch.to(self.device)
            out = model(device_batch)
            count += device_batch.energy.shape[0]
            if self.config["loss_per_atom"]:
                n_atoms = device_batch.num_atoms.view(-1)
                energy_pred = out.energy.view(-1) / n_atoms
                energy_target = device_batch.energy.view(-1) / n_atoms
                energy_loss = self.criterion(energy_pred, energy_target).detach().cpu().numpy()
            else:
                energy_loss = self.criterion(out.energy, device_batch.energy).detach().cpu().numpy()

            if bool(self.config["forces_weight"]):
                forces_count += device_batch.forces.shape[0]
                forces_loss = forces_criterion(out.forces, device_batch.forces).detach().cpu().numpy()
            else:
                forces_loss = 0.0
                
            if bool(self.config["virial_weight"]) and out.virial is not None and device_batch.virial is not None:
                virial_count += device_batch.virial.shape[0]
                if self.config["loss_per_atom"]:
                    n_atoms_v = device_batch.num_atoms.view(-1, 1, 1).expand_as(device_batch.virial)
                    virial_loss = self.criterion(out.virial / n_atoms_v, device_batch.virial / n_atoms_v).detach().cpu().numpy()
                else:
                    virial_loss = self.criterion(out.virial, device_batch.virial).detach().cpu().numpy()
            else:
                virial_loss = 0.0
                
            if bool(self.config["stress_weight"]) and out.stress is not None and device_batch.stress is not None:
                stress_count += device_batch.stress.shape[0]
                stress_loss = self.criterion(out.stress, device_batch.stress).detach().cpu().numpy()
            else:
                stress_loss = 0.0

            # use mean square loss here
            e_w = 1.0 - self.config["forces_weight"] - self.config["virial_weight"] - self.config["stress_weight"]
            total_loss = e_w * energy_loss + self.config["forces_weight"] * forces_loss + self.config["virial_weight"] * virial_loss + self.config["stress_weight"] * stress_loss
            running_loss += total_loss * device_batch.energy.shape[0]
            
            # energy errors
            if self.config["loss_per_atom"]:
                n_atoms_np = device_batch.num_atoms.detach().cpu().numpy()
                energy_targets = device_batch.energy.view(-1).detach().cpu().numpy() / n_atoms_np
                energy_outputs = out.energy.view(-1).detach().cpu().numpy() / n_atoms_np
            else:
                energy_targets = device_batch.energy.detach().cpu().numpy()
                energy_outputs = out.energy.detach().cpu().numpy()
            energy_running_ae += np.sum(np.abs(energy_targets - energy_outputs), axis=0)
            energy_running_se += np.sum(np.square(energy_targets - energy_outputs), axis=0)

            # force errors
            if bool(self.config["forces_weight"]) and out.forces is not None and device_batch.forces is not None:
                forces_targets = device_batch.forces.detach().cpu().numpy()
                forces_outputs = out.forces.detach().cpu().numpy()
                forces_diff = forces_targets - forces_outputs
                forces_running_c_ae += np.sum(np.abs(forces_diff))
                forces_running_c_se += np.sum(np.square(forces_diff))
            else:
                forces_running_c_ae = 0.0
                forces_running_c_se = 0.0
                
            # virial errors
            if bool(self.config["virial_weight"]) and out.virial is not None and device_batch.virial is not None:
                virial_targets = device_batch.virial.detach().cpu().numpy()
                virial_outputs = out.virial.detach().cpu().numpy()
                if self.config["loss_per_atom"]:
                    n_atoms_np = device_batch.num_atoms.detach().cpu().numpy()[:, np.newaxis, np.newaxis]
                    virial_targets = virial_targets / n_atoms_np
                    virial_outputs = virial_outputs / n_atoms_np
                virial_diff = virial_targets - virial_outputs
                virial_running_ae += np.sum(np.abs(virial_diff))
                virial_running_se += np.sum(np.square(virial_diff))
                
            # stress errors
            if bool(self.config["stress_weight"]) and out.stress is not None and device_batch.stress is not None:
                stress_targets = device_batch.stress.detach().cpu().numpy()
                stress_outputs = out.stress.detach().cpu().numpy()
                stress_diff = stress_targets - stress_outputs
                stress_running_ae += np.sum(np.abs(stress_diff))
                stress_running_se += np.sum(np.square(stress_diff))
        
        # Restore the model's original training state
        if was_training:
            model.train()

        energy_mae = energy_running_ae / count
        energy_rmse = np.sqrt(energy_running_se / count)

        if bool(self.config["forces_weight"]) and forces_count > 0:
            forces_mae = forces_running_c_ae / (forces_count * 3)
            forces_rmse = np.sqrt(forces_running_c_se / (forces_count * 3))
        else:
            forces_mae = 0
            forces_rmse = 0
            
        if bool(self.config["virial_weight"]) and virial_count > 0:
            virial_mae = virial_running_ae / (virial_count * 9)
            virial_rmse = np.sqrt(virial_running_se / (virial_count * 9))
        else:
            virial_mae = 0
            virial_rmse = 0
            
        if bool(self.config["stress_weight"]) and stress_count > 0:
            stress_mae = stress_running_ae / (stress_count * 9)
            stress_rmse = np.sqrt(stress_running_se / (stress_count * 9))
        else:
            stress_mae = 0
            stress_rmse = 0

        total_loss = running_loss / count

        evaluation = {
            "energy_mae": energy_mae,
            "energy_rmse": energy_rmse,
            "forces_mae": forces_mae,
            "forces_rmse": forces_rmse,
            "virial_mae": virial_mae,
            "virial_rmse": virial_rmse,
            "stress_mae": stress_mae,
            "stress_rmse": stress_rmse,
            "sqrt(total_loss)": np.sqrt(total_loss),
        }
        
        # Remove zeros from evaluation dict for unused metrics
        evaluation = {k: v for k, v in evaluation.items() if v != 0 or k == "sqrt(total_loss)"}

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
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt, datefmt)
        
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
        
        # determine max_steps or max_epochs
        if self.config["max_epochs"]:
            self.config["max_steps"] = None
            max_epochs = self.config["max_epochs"]
        else:
            max_epochs = int(self.config["max_steps"])
        
        # Log detailed model configuration and setup
        if self.rank == 0:
            logging.info("---------------- Configuration Settings ----------------")
            # Log for model
            logging.info(f"Model Type (model): {self.model_type}")
            logging.info(f"Number of Channels (num_channels): {self.config['num_channels']}")
            logging.info(f"Number of Layers (num_layers): {self.config['num_layers']}")
            logging.info(f"Cutoff Radius (cutoff): {self.config['cutoff']}")

            # Log for trainer
            logging.info(f"Validation Ratio (val_ratio): {self.config['val_ratio']}")
            logging.info(f"Batch Size (batch_size): {self.config['batch_size']}")
            logging.info(f"Learning Rate (learning_rate): {self.config['learning_rate']}")
            logging.info(f"Forces Weight (forces_weight): {self.config['forces_weight']}")
            if bool(self.config['stress_weight']):
                logging.info(f"Stress Weight (stress_weight): {self.config['stress_weight']}")
            if bool(self.config['virial_weight']):
                logging.info(f"Virial Weight (virial_weight): {self.config['virial_weight']}")
            logging.info(f"Loss per Atom (loss_per_atom): {self.config['loss_per_atom']}")
            
            if self.config["max_epochs"]:
                logging.info(f"Max Epochs (max_epochs): {max_epochs}")
            else:
                logging.info(f"Max Steps (max_steps): {self.config['max_steps']}")
            logging.info(f"Optimizer Type (optimizer_type): {self.optimizer_type}")
            if self.config['max_grad_norm']:
                logging.info(f"Gradient Clipping Norm (max_grad_norm): {self.config['max_grad_norm']}")
            logging.info(f"Log Interval (log_interval): {self.config['log_interval']}")
            logging.info(f"Early Stopping Patience (stop_patience): {self.config['stop_patience']}")
            logging.info(f"Scheduler Type (scheduler_type): {self.config['scheduler_type']}")

            # Log for data
            logging.info(f"Random Seed (random_seed): {self.config['random_seed']}")
            if self.config['save_split']:
                logging.info(f"Save Split File Name (save_split): {self.config['save_split']}")
            if self.config['load_split']:
                logging.info(f"Load Split File Name (load_split): {self.config['load_split']}")
            if self.config['norm_data'] and not self.config['norm_per_atom']:
                logging.info(f"Data Normalization (norm_data): {self.config['norm_data']}")
            if self.config['norm_per_atom']:
                logging.info(f"Data Normalization per Atom (norm_per_atom): {self.config['norm_per_atom']}")
            if not self.config['norm_data'] and not self.config['norm_per_atom']:
                logging.info("Data Normalization (norm_data): False")
            
            # Log for DDP
            logging.info(f"Distributed Training (distributed): {self.distributed}")
            if self.distributed:
                logging.info(f"Master Port (master_port): {self.config['master_port']}")
                logging.info(f"Distributed Timeout (dist_timeout) (s): {self.config['dist_timeout']}")
            
            # Log for output
            logging.info(f"Output Directory (output_dir): {self.config['output_dir']}")
            logging.info(f"Output Log File (output_log): {self.config['output_log']}")
            logging.info(f"Output Model File (output_model): {self.config['output_model']}")

            # Log input config after logging is set up
            if self.config["log_input"]:
                logging.info(f"Input config: {self.input_config}")

            # To do: log all default model parameters
            if self.config['debug']:
                logging.debug(f"Debug Mode (debug): {self.config['debug']}")
                logging.debug(f"All parameters: {self.config}") # log all default parameters except for other model default parameters
        
        # Initialize counters
        local_steps = 0
        total_steps = 0
        running_loss = 0.0
        running_loss_count = 0
        training_time = 0.0
        prev_loss = None
        best_val_loss = np.inf
        
        if self.rank == 0:
            logging.info("---------------------- Training ------------------------")
        # Training loop
        for epoch in range(max_epochs):
            if epoch >= max_epochs - 1:
                logging.info(f"Reached maximum epochs ({max_epochs}), stopping training!")
                if self.distributed:
                    self._cleanup_distributed()
                return
            
            if hasattr(self.train_sampler, "set_epoch"):
                # For distributed sampler, set epoch for shuffling
                self.train_sampler.set_epoch(epoch)
                
            self.model.train()
            for batch in self.train_loader:
                train_start_time = time.time()
                
                device_batch = batch.to(self.device)
                
                # Forward pass and backward pass
                self.optimizer.zero_grad()
                out = self.model(device_batch)
                
                # Calculate losses
                if self.config["loss_per_atom"]:
                    n_atoms = device_batch.num_atoms.view(-1)
                    energy_pred = out.energy.view(-1) / n_atoms
                    energy_target = device_batch.energy.view(-1) / n_atoms
                    energy_loss = self.criterion(energy_pred, energy_target)
                else:
                    energy_loss = self.criterion(out.energy, device_batch.energy)
                if bool(self.config["forces_weight"]):
                    forces_loss = forces_criterion(out.forces, device_batch.forces)
                else:
                    forces_loss = 0.0
                
                if bool(self.config["virial_weight"]) and out.virial is not None and device_batch.virial is not None:
                    if self.config["loss_per_atom"]:
                        n_atoms_v = device_batch.num_atoms.view(-1, 1, 1).expand_as(device_batch.virial)
                        virial_loss = self.criterion(out.virial / n_atoms_v, device_batch.virial / n_atoms_v)
                    else:
                        virial_loss = self.criterion(out.virial, device_batch.virial)
                else:
                    virial_loss = 0.0
                    
                if bool(self.config["stress_weight"]) and out.stress is not None and device_batch.stress is not None:
                    stress_loss = self.criterion(out.stress, device_batch.stress)
                else:
                    stress_loss = 0.0
                
                # Total loss
                e_w = 1.0 - self.config["forces_weight"] - self.config["virial_weight"] - self.config["stress_weight"]
                total_loss = e_w * energy_loss + self.config["forces_weight"] * forces_loss + self.config["virial_weight"] * virial_loss + self.config["stress_weight"] * stress_loss
                total_loss.backward()
                
                # Apply gradient clipping
                if self.config.get("max_grad_norm") is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get("max_grad_norm"))

                self.optimizer.step()

                # Update running loss
                running_loss += total_loss.detach().cpu().numpy() * device_batch.energy.shape[0]
                running_loss_count += device_batch.energy.shape[0]
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
                    
                    if self.scheduler_type=="ReduceLROnPlateau":
                        self.scheduler.step(smooth_loss)
                    
                    # Check early stopping and save best model
                    if not self.early_stop(math.sqrt(smooth_loss), best_val_loss):
                        best_val_loss = math.sqrt(smooth_loss)
                        
                        self._save_model(self.config["output_model"], total_steps, best_val_loss)
                    else:
                        logging.info(f"Early stopping, training complete")
                        
                        if self.distributed:
                            self._cleanup_distributed()
                        return
                
                # Count steps
                local_steps += 1
                
                if self.scheduler_type!="ReduceLROnPlateau":
                    self.scheduler.step() # update learning rate based on scheduler type
                
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

    if bool(args.model_type):
        args.model_type = args.model_type
    else:
        args.model_type = "painn"
        
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
            mp.spawn(
                process_function,
                args=(num_gpus, args.model_type, config, args.dataset),
                nprocs=num_gpus,
                join=True
            )
        else:
            # Single-GPU or CPU training
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
