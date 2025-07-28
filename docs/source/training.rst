Training Guide
=============

This guide covers how to train IANN models for energy and force prediction.

Preparing Your Dataset
----------------------

IANN works with ASE database (.db) or trajectory (.traj) files. Your data should contain:

* Atomic structures (positions, atomic numbers)
* Energy labels
* Force labels (optional, but recommended)


Running Training 
----------------

Here is an simple example ``train.py`` of how to run training:

.. code-block:: python

   from iann.trainer import Trainer

   # Define the trainer
   trainer = Trainer(
      model="painn",
      config={'device': 'cuda', 
              'output_dir': 'output',
              'output_log': 'output.log',
              'output_model': 'model.pt'},
      distributed=False,
      )

   # Train the model
   trainer.train("dataset.traj")

Available models for ``model``:

* painn
* nequip
* mace
* equiformerV2

Available configurations for ``config``:

.. code-block:: python

   config = {
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
   "load_model": False, # load model from checkpoint
   "max_steps": 1000000, # maximum number of steps
   "max_epochs": None,  # None if setup max_steps, otherwise max_epochs
   "optimizer_type": "adam", # optimizer type: "adam", "sgd", "rmsprop", "adagrad", etc.
   "max_grad_norm": None,    # gradient clipping norm
   "log_interval": 2000, # log interval
   "stop_patience": 200, # patience for early stopping
   "scheduler_type": "LambdaLR", # scheduler type: "ReduceLROnPlateau", "LambdaLR", etc.
   # parameters for data
   "random_seed": 666, # random seed for reproducibility
   "save_split": False, # save split file name
   "load_split": False, # load split file name
   "norm_data": False, # normalize data
   "norm_per_atom": False, # normalize data per atom
   # parameters for DDP (Parallelization)
   "dist_timeout": 600,  # timeout (seconds) for distributed operations
   "master_port": 12356, # port for distributed operations
   # parameters for output
   "output_dir": "output", # output directory
   "output_log": "output.log", # log file
   "output_model": "model.pt", # model file
   "log_input": False, # log your costomized input config
   "debug": False, # debug mode
   }

.. note::
   There are more adjustable parameters for each model, please refer to the :doc:`api` reference for details, or check the source code for more details (all adjustable parameters are passed as `kwargs.get` in the model class).

Directly run the training script in command line:

.. code-block:: bash

   # Run on a local machine
   python train.py


It will generate a log file and a checkpoint file in the output directory. The log file will record the training progress. The checkpoint file will record the model parameters. 
The example log file is shown below:

.. code-block:: text

   2025-07-25 16:28:55 [RANK0] [INFO ]  PyTorch version: 2.4.0
   2025-07-25 16:28:55 [RANK0] [INFO ]  Running in single-GPU Node login12
   2025-07-25 16:28:55 [RANK0] [INFO ]  Hardware architecture: NVIDIA A100-PCIE-40GB
   2025-07-25 16:28:55 [RANK0] [INFO ]  Loading data from dataset.traj
   2025-07-25 16:28:56 [RANK0] [INFO ]  Dataset size: 12974, training set size: 11676, validation set size: 1298
   2025-07-25 16:28:56 [RANK0] [INFO ]  Compute forces: True
   2025-07-25 16:28:58 [RANK0] [INFO ]  Total trainable parameters: 87081
   2025-07-25 16:28:58 [RANK0] [INFO ]  Total memory of the model: 0.33 MB
   2025-07-25 16:28:58 [RANK0] [INFO ]  ---------------- Configuration Settings ----------------
   2025-07-25 16:28:58 [RANK0] [INFO ]  Model Type (model): painn
   2025-07-25 16:28:58 [RANK0] [INFO ]  Number of Channels (num_channels): 128
   2025-07-25 16:28:58 [RANK0] [INFO ]  Number of Layers (num_layers): 2
   2025-07-25 16:28:58 [RANK0] [INFO ]  Cutoff Radius (cutoff): 5.5
   2025-07-25 16:28:58 [RANK0] [INFO ]  Validation Ratio (val_ratio): 0.1
   2025-07-25 16:28:58 [RANK0] [INFO ]  Batch Size (batch_size): 12
   2025-07-25 16:28:58 [RANK0] [INFO ]  Learning Rate (learning_rate): 0.0001
   2025-07-25 16:28:58 [RANK0] [INFO ]  Forces Weight (forces_weight): 0.9
   2025-07-25 16:28:58 [RANK0] [INFO ]  Max Steps (max_steps): 1000000
   2025-07-25 16:28:58 [RANK0] [INFO ]  Optimizer Type (optimizer_type): adam
   2025-07-25 16:28:58 [RANK0] [INFO ]  Log Interval (log_interval): 100
   2025-07-25 16:28:58 [RANK0] [INFO ]  Early Stopping Patience (stop_patience): 200
   2025-07-25 16:28:58 [RANK0] [INFO ]  Scheduler Type (scheduler_type): LambdaLR
   2025-07-25 16:28:58 [RANK0] [INFO ]  Random Seed (random_seed): 666
   2025-07-25 16:28:58 [RANK0] [INFO ]  Data Normalization (norm_data): False
   2025-07-25 16:28:58 [RANK0] [INFO ]  Distributed Training (distributed): False
   2025-07-25 16:28:58 [RANK0] [INFO ]  Output Directory (output_dir): output
   2025-07-25 16:28:58 [RANK0] [INFO ]  Output Log File (output_log): output.log
   2025-07-25 16:28:58 [RANK0] [INFO ]  Output Model File (output_model): model.pt
   2025-07-25 16:28:58 [RANK0] [INFO ]  ---------------------- Training ------------------------
   2025-07-25 16:29:06 [RANK0] [INFO ]  step=0, energy_mae=572.917, energy_rmse=662.639, forces_mae=0.628, forces_rmse=2.553, sqrt(total_loss)=209.548, sqrt(train_loss)=200.186, patience=  0, training time=0.006 min, eval time=0.133 min
   2025-07-25 16:29:25 [RANK0] [INFO ]  step=100, energy_mae=113.896, energy_rmse=128.368, forces_mae=2.170, forces_rmse=4.145, sqrt(total_loss)=40.646, sqrt(train_loss)=170.663, patience=  0, training time=0.063 min, eval time=0.135 min
   2025-07-25 16:29:45 [RANK0] [INFO ]  step=200, energy_mae=26.501, energy_rmse=37.474, forces_mae=2.508, forces_rmse=4.613, sqrt(total_loss)=12.055, sqrt(train_loss)=15.987, patience=  0, training time=0.082 min, eval time=0.143 min
   2025-07-25 16:30:05 [RANK0] [INFO ]  step=300, energy_mae=24.298, energy_rmse=32.627, forces_mae=2.132, forces_rmse=4.113, sqrt(total_loss)=10.517, sqrt(train_loss)=12.032, patience=  0, training time=0.060 min, eval time=0.147 min
   2025-07-25 16:30:24 [RANK0] [INFO ]  step=400, energy_mae=22.511, energy_rmse=29.405, forces_mae=1.882, forces_rmse=3.756, sqrt(total_loss)=9.493, sqrt(train_loss)=10.354, patience=  0, training time=0.070 min, eval time=0.135 min
   2025-07-25 16:30:45 [RANK0] [INFO ]  step=500, energy_mae=17.992, energy_rmse=24.286, forces_mae=1.810, forces_rmse=3.567, sqrt(total_loss)=7.902, sqrt(train_loss)=8.628, patience=  0, training time=0.059 min, eval time=0.152 min
   2025-07-25 16:31:04 [RANK0] [INFO ]  step=600, energy_mae=13.559, energy_rmse=19.045, forces_mae=1.970, forces_rmse=3.741, sqrt(total_loss)=6.324, sqrt(train_loss)=6.984, patience=  0, training time=0.064 min, eval time=0.144 min
   2025-07-25 16:31:24 [RANK0] [INFO ]  step=700, energy_mae=10.003, energy_rmse=15.537, forces_mae=2.198, forces_rmse=4.348, sqrt(total_loss)=5.315, sqrt(train_loss)=6.298, patience=  0, training time=0.076 min, eval time=0.147 min
   2025-07-25 16:31:42 [RANK0] [INFO ]  step=800, energy_mae=8.993, energy_rmse=14.047, forces_mae=2.101, forces_rmse=4.249, sqrt(total_loss)=4.861, sqrt(train_loss)=5.021, patience=  0, training time=0.062 min, eval time=0.122 min
   2025-07-25 16:32:01 [RANK0] [INFO ]  step=900, energy_mae=9.039, energy_rmse=13.523, forces_mae=1.853, forces_rmse=3.856, sqrt(total_loss)=4.660, sqrt(train_loss)=4.602, patience=  0, training time=0.063 min, eval time=0.153 min
   2025-07-25 16:32:21 [RANK0] [INFO ]  step=1000, energy_mae=7.431, energy_rmse=11.518, forces_mae=1.709, forces_rmse=3.661, sqrt(total_loss)=4.051, sqrt(train_loss)=4.473, patience=  0, training time=0.080 min, eval time=0.128 min
   2025-07-25 16:32:40 [RANK0] [INFO ]  step=1100, energy_mae=6.375, energy_rmse=10.158, forces_mae=1.708, forces_rmse=3.811, sqrt(total_loss)=3.666, sqrt(train_loss)=4.162, patience=  0, training time=0.079 min, eval time=0.132 min
   2025-07-25 16:32:58 [RANK0] [INFO ]  step=1200, energy_mae=5.888, energy_rmse=9.439, forces_mae=1.600, forces_rmse=3.639, sqrt(total_loss)=3.438, sqrt(train_loss)=3.475, patience=  0, training time=0.063 min, eval time=0.122 min

Multi-GPU Training examples
--------------------------

Here is an example of how to run multi-GPU training on NERSC:

.. code-block:: bash

   #!/bin/bash
   #SBATCH -N 2                   # Number of nodes
   #SBATCH -C gpu                 # Use GPU nodes
   #SBATCH -q debug               # Use regular/debug queue
   #SBATCH -t 00:30:00            # Time limit
   #SBATCH -A m2997               # Your account
   #SBATCH --gpus-per-node=4      # GPUs per node
   #SBATCH --ntasks-per-node=4    # Number of tasks per node
   #SBATCH --cpus-per-task=1      # Number of CPUs per task

   # Load environments, such as:
   export PYTHONPATH=/pscratch/sd/c/changzhi/softwares/IANN_v2/IANN/:$PYTHONPATH
   module purge
   module load PrgEnv-nvidia; module load openmpi
   
   # NERSC specific environment variables for parallelization
   export FI_CXI_RDZV_GET_MIN=0 # vender bugs fixed on nersc for multiple nodes
   export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216 # vender bugs fixed on nersc

   # GPUs per node and number of nodes
   export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
   export NNODES=$SLURM_NNODES

   # Run the training script on multiple GPUs/CPUs
   srun -N $NNODES -n $((NNODES*GPUS_PER_NODE)) python train.py


Similarly, the log file and checkpoint file will be saved in the output directory. The example log file is shown below:

.. code-block:: text

   2025-07-25 17:12:07 [RANK0] [INFO ]  PyTorch version: 2.4.0
   2025-07-25 17:12:07 [RANK0] [INFO ]  Node List: nid[008436,008509]
   2025-07-25 17:12:07 [RANK0] [INFO ]  World Size (number of GPUs): 8
   2025-07-25 17:12:07 [RANK0] [INFO ]  Master Address: nid008436
   2025-07-25 17:12:07 [RANK0] [INFO ]  Master Port: 12356
   2025-07-25 17:12:08 [RANK0] [INFO ]  Process 0 using device cuda:0 on nid008436. GPU architecture: NVIDIA A100-SXM4-80GB
   2025-07-25 17:12:08 [RANK1] [INFO ]  Process 1 using device cuda:1 on nid008509. GPU architecture: NVIDIA A100-SXM4-80GB
   2025-07-25 17:12:08 [RANK2] [INFO ]  Process 2 using device cuda:2 on nid008436. GPU architecture: NVIDIA A100-SXM4-80GB
   2025-07-25 17:12:08 [RANK3] [INFO ]  Process 3 using device cuda:3 on nid008509. GPU architecture: NVIDIA A100-SXM4-80GB
   2025-07-25 17:12:08 [RANK4] [INFO ]  Process 4 using device cuda:0 on nid008436. GPU architecture: NVIDIA A100-SXM4-80GB
   2025-07-25 17:12:08 [RANK5] [INFO ]  Process 5 using device cuda:1 on nid008509. GPU architecture: NVIDIA A100-SXM4-80GB
   2025-07-25 17:12:08 [RANK6] [INFO ]  Process 6 using device cuda:2 on nid008436. GPU architecture: NVIDIA A100-SXM4-80GB
   2025-07-25 17:12:08 [RANK7] [INFO ]  Process 7 using device cuda:3 on nid008509. GPU architecture: NVIDIA A100-SXM4-80GB
   2025-07-25 17:12:09 [RANK0] [INFO ]  Loading data from dataset.traj
   2025-07-25 17:12:09 [RANK0] [INFO ]  Dataset size: 12974, training set size: 11676, validation set size: 1298
   2025-07-25 17:12:09 [RANK0] [INFO ]  Compute forces: True
   2025-07-25 17:12:22 [RANK0] [INFO ]  Total trainable parameters: 87081
   2025-07-25 17:12:22 [RANK0] [INFO ]  Total memory of the model: 0.33 MB
   2025-07-25 17:12:22 [RANK0] [INFO ]  ---------------- Configuration Settings ----------------
   2025-07-25 17:12:22 [RANK0] [INFO ]  Model Type (model): painn
   2025-07-25 17:12:22 [RANK0] [INFO ]  Number of Channels (num_channels): 128
   2025-07-25 17:12:22 [RANK0] [INFO ]  Number of Layers (num_layers): 2
   2025-07-25 17:12:22 [RANK0] [INFO ]  Cutoff Radius (cutoff): 5.5
   2025-07-25 17:12:22 [RANK0] [INFO ]  Validation Ratio (val_ratio): 0.1
   2025-07-25 17:12:22 [RANK0] [INFO ]  Batch Size (batch_size): 12
   2025-07-25 17:12:22 [RANK0] [INFO ]  Learning Rate (learning_rate): 0.0001
   2025-07-25 17:12:22 [RANK0] [INFO ]  Forces Weight (forces_weight): 0.9
   2025-07-25 17:12:22 [RANK0] [INFO ]  Max Steps (max_steps): 1000000
   2025-07-25 17:12:22 [RANK0] [INFO ]  Optimizer Type (optimizer_type): adam
   2025-07-25 17:12:22 [RANK0] [INFO ]  Log Interval (log_interval): 100
   2025-07-25 17:12:22 [RANK0] [INFO ]  Early Stopping Patience (stop_patience): 200
   2025-07-25 17:12:22 [RANK0] [INFO ]  Scheduler Type (scheduler_type): LambdaLR
   2025-07-25 17:12:22 [RANK0] [INFO ]  Random Seed (random_seed): 666
   2025-07-25 17:12:22 [RANK0] [INFO ]  Data Normalization (norm_data): False
   2025-07-25 17:12:22 [RANK0] [INFO ]  Distributed Training (distributed): True
   2025-07-25 17:12:22 [RANK0] [INFO ]  Master Port (master_port): 12356
   2025-07-25 17:12:22 [RANK0] [INFO ]  Distributed Timeout (dist_timeout) (s): 600
   2025-07-25 17:12:22 [RANK0] [INFO ]  Output Directory (output_dir): output
   2025-07-25 17:12:22 [RANK0] [INFO ]  Output Log File (output_log): output.log
   2025-07-25 17:12:22 [RANK0] [INFO ]  Output Model File (output_model): model.pt
   2025-07-25 17:12:22 [RANK0] [INFO ]  ---------------------- Training ------------------------
   2025-07-25 17:12:24 [RANK0] [INFO ]  step=0, energy_mae=551.066, energy_rmse=626.241, forces_mae=0.573, forces_rmse=0.838, sqrt(total_loss)=198.037, sqrt(train_loss)=155.493, patience=  0, training time=0.016 min, eval time=0.010 min
   2025-07-25 17:12:27 [RANK0] [INFO ]  step=200, energy_mae=507.809, energy_rmse=576.979, forces_mae=0.570, forces_rmse=0.819, sqrt(total_loss)=182.460, sqrt(train_loss)=200.346, patience=  0, training time=0.029 min, eval time=0.008 min
   2025-07-25 17:12:30 [RANK0] [INFO ]  step=400, energy_mae=442.696, energy_rmse=501.847, forces_mae=0.658, forces_rmse=0.936, sqrt(total_loss)=158.702, sqrt(train_loss)=185.743, patience=  0, training time=0.027 min, eval time=0.008 min
   2025-07-25 17:12:33 [RANK0] [INFO ]  step=600, energy_mae=306.090, energy_rmse=343.443, forces_mae=1.116, forces_rmse=1.694, sqrt(total_loss)=108.616, sqrt(train_loss)=151.850, patience=  0, training time=0.028 min, eval time=0.008 min
   2025-07-25 17:12:36 [RANK0] [INFO ]  step=800, energy_mae=27.989, energy_rmse=41.629, forces_mae=3.050, forces_rmse=4.935, sqrt(total_loss)=13.392, sqrt(train_loss)=70.639, patience=  0, training time=0.027 min, eval time=0.008 min
   2025-07-25 17:12:39 [RANK0] [INFO ]  step=1000, energy_mae=34.184, energy_rmse=42.730, forces_mae=2.550, forces_rmse=4.071, sqrt(total_loss)=13.698, sqrt(train_loss)=15.438, patience=  0, training time=0.027 min, eval time=0.008 min
   2025-07-25 17:12:42 [RANK0] [INFO ]  step=1200, energy_mae=25.578, energy_rmse=35.079, forces_mae=2.552, forces_rmse=4.081, sqrt(total_loss)=11.319, sqrt(train_loss)=14.864, patience=  0, training time=0.027 min, eval time=0.008 min
   2025-07-25 17:12:45 [RANK0] [INFO ]  step=1400, energy_mae=25.029, energy_rmse=32.945, forces_mae=2.389, forces_rmse=3.819, sqrt(total_loss)=10.643, sqrt(train_loss)=12.152, patience=  0, training time=0.026 min, eval time=0.008 min
   2025-07-25 17:12:48 [RANK0] [INFO ]  step=1600, energy_mae=23.706, energy_rmse=31.331, forces_mae=2.306, forces_rmse=3.685, sqrt(total_loss)=10.136, sqrt(train_loss)=11.054, patience=  0, training time=0.026 min, eval time=0.008 min
   2025-07-25 17:12:51 [RANK0] [INFO ]  step=1800, energy_mae=23.139, energy_rmse=30.280, forces_mae=2.203, forces_rmse=3.517, sqrt(total_loss)=9.800, sqrt(train_loss)=10.503, patience=  0, training time=0.026 min, eval time=0.008 min
   2025-07-25 17:12:54 [RANK0] [INFO ]  step=2000, energy_mae=21.993, energy_rmse=29.099, forces_mae=2.135, forces_rmse=3.405, sqrt(total_loss)=9.429, sqrt(train_loss)=11.475, patience=  0, training time=0.026 min, eval time=0.008 min
   2025-07-25 17:12:57 [RANK0] [INFO ]  step=2200, energy_mae=21.388, energy_rmse=27.886, forces_mae=2.014, forces_rmse=3.204, sqrt(total_loss)=9.041, sqrt(train_loss)=11.517, patience=  0, training time=0.028 min, eval time=0.008 min
   2025-07-25 17:13:00 [RANK0] [INFO ]  step=2400, energy_mae=20.517, energy_rmse=26.787, forces_mae=1.953, forces_rmse=3.099, sqrt(total_loss)=8.695, sqrt(train_loss)=9.771, patience=  0, training time=0.027 min, eval time=0.008 min



Here is an example of how to run multi-GPU training on S3DF:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=train
   #SBATCH --nodes=2
   #SBATCH --tasks-per-node=1
   #SBATCH --cpus-per-task=1
   #SBATCH --gpus-per-node=1
   #SBATCH --time=00:30:00
   #SBATCH --partition=ampere
   #SBATCH --account=suncat:normal

   # Load environments, such as:
   conda activate /sdf/home/c/changzhi/softwares/anoconda3/envs/painn
   export PYTHONPATH=/sdf/home/c/changzhi/changzhi/softwares/IANN_v2/IANN:$PYTHONPATH

   # GPUs per node and number of nodes
   export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
   export NNODES=$SLURM_NNODES

   # Run the training script on multiple GPUs/CPUs
   srun -N $NNODES -n $((NNODES*GPUS_PER_NODE)) python train.py


See the :doc:`parallelization` guide for details on distributed training.



Continuous Training
-------------------  

If you want to continue training from a previous checkpoint, you can use the ``load_model`` option.

.. code-block:: python
   
   from iann.trainer import Trainer

   # Load the model from a checkpoint and continue training
   trainer = Trainer(
      model="painn",
      config={"device": 'cuda',
               'load_model': '/path/to/model.pt', # path to the model checkpoint
               'output_dir': 'output',
               'output_model': 'model.pt'},
      distributed=False
   )

Only add ``'load_model': /path/to/model.pt`` and not change anything else if you want to continue training from a previous checkpoint.

.. note::
   If you want to continue training from a previous checkpoint, you need to use the same model type and model parameters as the original checkpoint.


Parameters Explanation
----------------------
Here is a list of default parameters and their explanations in ``config``:

* ``num_channels``: number of channels in the model
* ``num_layers``: number of layers in the model
* ``cutoff``: cutoff radius in the model
* ``device``: device to run the training on, e.g. `cuda` or `cpu`
* ``val_ratio``: validation set ratio
* ``batch_size``: batch size
* ``learning_rate``: initial learning rate
* ``forces_weight``: weight of the force loss. calculate forces if ``forces_weight > 0``
* ``load_model``: path to the model checkpoint
* ``max_steps``: maximum number of steps
* ``max_epochs``: maximum number of epochs
* ``optimizer_type``: optimizer type: "adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "adamw"
* ``max_grad_norm``: maximum gradient norm for gradient clipping
* ``log_interval``: log interval for training progress
* ``stop_patience``: patience for early stopping
* ``scheduler_type``: scheduler type: "ReduceLROnPlateau", "LambdaLR", "CosineAnnealingLR", "CosineAnnealingWarmRestarts", "StepLR", "MultiStepLR", "ExponentialLR"
* ``random_seed``: random seed for reproducibility
* ``save_split``: whether to save the train/validation split to a file
* ``load_split``: path to load a pre-defined train/validation split file
* ``norm_data``: whether to normalize the data
* ``norm_per_atom``: whether to normalize data per atom
* ``dist_timeout``: timeout (seconds) for distributed operations
* ``master_port``: master port for distributed training
* ``output_dir``: output directory
* ``output_log``: output log file name
* ``output_model``: output model file name
* ``log_input``: whether to log your costomized input config
* ``debug``: whether to use debug mode

There are more adjustable parameters for each model, please refer to the :doc:`api` reference for details, or check the source code for more details (all adjustable parameters are passed as `kwargs.get` in the model class).

.. note::
   Choose either ``max_steps`` or ``max_epochs`` to setup the training process. If both are set, the ``max_steps`` will be ignored. Similarly, for ``norm_data`` and ``norm_per_atom``, if both are set, the ``norm_data`` will be ignored.

Monitoring Training
-------------------

Training progress is logged to the output directory. You can monitor:

* Energy and force prediction errors
* Training and validation losses
* Model checkpoints


Training Tips
-------------

1. **Data Preparation**

   * Ensure your dataset is properly normalized
   * Include diverse structures for better generalization
   * Balance the dataset if possible

2. **Model Configuration**

   * Start with a small model and increase size if needed
   * Use appropriate cutoff radius for your system
   * Adjust forces_weight based on your priorities

3. **Training Process**

   * Monitor validation loss for early stopping
   * Use learning rate scheduling if needed
   * Consider gradient clipping for stability

4. **Performance Optimization**

   * Use the largest batch size that fits in memory
   * Enable mixed precision training if available
   * Profile your training to identify bottlenecks

For more advanced training options and troubleshooting, see the :doc:`api` reference. 