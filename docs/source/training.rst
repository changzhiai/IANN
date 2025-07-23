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

   from iann.trainer.trainer import Trainer

   trainer = Trainer(
      model="painn",
      config={'device': 'cuda', 
               'output_dir': 'output',
               'output_model': 'model.pt'},
      distributed=False
      )
   trainer.train("dataset.traj")

Available models for ``model``:

* painn
* nequip
* mace
* equiformerV2

Available configurations for ``config``:

.. code-block:: python

   config = {
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
      'output_log': 'output.log',
      "max_grad_norm": None,    # Gradient clipping norm
      "output_model": "model.pt",
   }


Directly run the training script in command line:

.. code-block:: bash

   # Run on a local machine
   python train.py


It will generate a log file and a checkpoint file in the output directory. The log file will record the training progress. The checkpoint file will record the model parameters. 
The example log file is shown below:

.. code-block:: text

   2025-07-21 12:52:09,282 [RANK0] [INFO ]  PyTorch version: 2.4.0
   2025-07-21 12:52:09,284 [RANK0] [INFO ]  Node List: nid[008380-008381]
   2025-07-21 12:52:09,284 [RANK0] [INFO ]  World Size (number of GPUs): 8
   2025-07-21 12:52:09,284 [RANK0] [INFO ]  Master Address: nid008380
   2025-07-21 12:52:09,284 [RANK0] [INFO ]  Master Port: 12356
   2025-07-21 12:52:09,454 [RANK0] [INFO ]  Process 0 using device cuda:0 on nid008380. GPU architecture: NVIDIA A100-SXM4-80GB
   2025-07-21 12:52:11,024 [RANK0] [INFO ]  Loading data from ~/dataset.traj
   2025-07-21 12:52:11,028 [RANK0] [INFO ]  Dataset size: 12974, training set size: 11676, validation set size: 1298
   2025-07-21 12:53:04,407 [RANK0] [INFO ]  Total trainable parameters: 821089
   2025-07-21 12:53:04,408 [RANK0] [INFO ]  Total memory: 3.13 MB
   2025-07-21 12:53:04,524 [RANK0] [INFO ]  ---------------- Configuration Settings ----------------
   2025-07-21 12:53:04,527 [RANK0] [INFO ]  Model Type (model): equiformerv2
   2025-07-21 12:53:04,527 [RANK0] [INFO ]  Node Size (node_size): 16
   2025-07-21 12:53:04,527 [RANK0] [INFO ]  Number of Interactions (num_interactions): 3
   2025-07-21 12:53:04,527 [RANK0] [INFO ]  Cutoff Radius (cutoff): 5.5
   2025-07-21 12:53:04,527 [RANK0] [INFO ]  Forces Weight (forces_weight): 0.0
   2025-07-21 12:53:04,527 [RANK0] [INFO ]  Normalization (normalization): True
   2025-07-21 12:53:04,527 [RANK0] [INFO ]  Atomwise Normalization (atomwise_normalization): False
   2025-07-21 12:53:04,527 [RANK0] [INFO ]  Batch Size (batch_size): 24
   2025-07-21 12:53:04,527 [RANK0] [INFO ]  Initial Learning Rate (initial_lr): 0.0001
   2025-07-21 12:53:04,528 [RANK0] [INFO ]  Max Steps (max_steps): 1000000
   2025-07-21 12:53:04,528 [RANK0] [INFO ]  Max Epochs (max_epochs): None
   2025-07-21 12:53:04,528 [RANK0] [INFO ]  Early Stopping Patience (stop_patience): 200
   2025-07-21 12:53:04,528 [RANK0] [INFO ]  Plateau Scheduler (plateau_scheduler): False
   2025-07-21 12:53:04,528 [RANK0] [INFO ]  Validation Ratio (val_ratio): 0.1
   2025-07-21 12:53:04,528 [RANK0] [INFO ]  Split File (split_file): None
   2025-07-21 12:53:04,528 [RANK0] [INFO ]  Target Mean (target_mean): -594.789978
   2025-07-21 12:53:04,528 [RANK0] [INFO ]  Target Stddev (target_stddev): 361.503448
   2025-07-21 12:53:04,528 [RANK0] [INFO ]  Distributed Training (distributed): True
   2025-07-21 12:53:04,528 [RANK0] [INFO ]  Master Port (master_port): 12356
   2025-07-21 12:53:04,528 [RANK0] [INFO ]  Distributed Timeout (dist_timeout) (s): 600
   2025-07-21 12:53:04,528 [RANK0] [INFO ]  ---------------------- Training ------------------------
   2025-07-21 12:53:07,976 [RANK0] [INFO ]  step=186000, energy_mae=7.901, energy_rmse=9.284, forces_mae=0.000, forces_rmse=0.000, sqrt(total_loss)=9.284, sqrt(train_loss)=1.632, patience=  0, training time=0.022 min, eval time=0.035 min
   2025-07-21 12:54:36,354 [RANK0] [INFO ]  step=187000, energy_mae=1.614, energy_rmse=2.418, forces_mae=0.000, forces_rmse=0.000, sqrt(total_loss)=2.418, sqrt(train_loss)=3.668, patience=  0, training time=1.293 min, eval time=0.034 min
   2025-07-21 12:56:07,311 [RANK0] [INFO ]  step=188000, energy_mae=2.025, energy_rmse=2.903, forces_mae=0.000, forces_rmse=0.000, sqrt(total_loss)=2.903, sqrt(train_loss)=3.530, patience=  0, training time=1.336 min, eval time=0.035 min
   2025-07-21 12:57:36,083 [RANK0] [INFO ]  step=189000, energy_mae=1.612, energy_rmse=2.484, forces_mae=0.000, forces_rmse=0.000, sqrt(total_loss)=2.484, sqrt(train_loss)=3.607, patience=  0, training time=1.300 min, eval time=0.034 min
   2025-07-21 12:59:05,271 [RANK0] [INFO ]  step=190000, energy_mae=2.149, energy_rmse=3.215, forces_mae=0.000, forces_rmse=0.000, sqrt(total_loss)=3.215, sqrt(train_loss)=3.399, patience=  0, training time=1.306 min, eval time=0.035 min
   2025-07-21 13:00:35,395 [RANK0] [INFO ]  step=191000, energy_mae=1.584, energy_rmse=2.808, forces_mae=0.000, forces_rmse=0.000, sqrt(total_loss)=2.808, sqrt(train_loss)=3.430, patience=  1, training time=1.322 min, eval time=0.034 min
   2025-07-21 13:02:04,686 [RANK0] [INFO ]  step=192000, energy_mae=2.003, energy_rmse=2.728, forces_mae=0.000, forces_rmse=0.000, sqrt(total_loss)=2.728, sqrt(train_loss)=3.549, patience=  1, training time=1.307 min, eval time=0.034 min
   2025-07-21 13:03:34,198 [RANK0] [INFO ]  step=193000, energy_mae=1.735, energy_rmse=2.535, forces_mae=0.000, forces_rmse=0.000, sqrt(total_loss)=2.535, sqrt(train_loss)=3.320, patience=  1, training time=1.311 min, eval time=0.034 min
   2025-07-21 13:05:03,454 [RANK0] [INFO ]  step=194000, energy_mae=2.091, energy_rmse=3.040, forces_mae=0.000, forces_rmse=0.000, sqrt(total_loss)=3.040, sqrt(train_loss)=3.321, patience=  1, training time=1.308 min, eval time=0.034 min
   2025-07-21 13:06:32,814 [RANK0] [INFO ]  step=195000, energy_mae=1.648, energy_rmse=2.716, forces_mae=0.000, forces_rmse=0.000, sqrt(total_loss)=2.716, sqrt(train_loss)=3.531, patience=  2, training time=1.309 min, eval time=0.034 min


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

   trainer = Trainer(
      model="painn",
      config={"device": 'cuda',
               'load_model': '/path/to/model.pt', # path to the model checkpoint
               'output_dir': 'output',
               'output_model': 'model.pt'},
      distributed=False
   )

Only add ``'load_model': /path/to/model.pt`` and not change anything else if you want to continue training from a previous checkpoint.


Parameters Explanation
----------------------
Here is a list of default parameters and their explanations in ``config``:

* ``device``: device to run the training on, e.g. `cuda` or `cpu`
* ``load_model``: path to the model checkpoint
* ``output_dir``: output directory
* ``output_model``: output model file name
* ``distributed``: whether to use distributed training
* ``node_size``: number of nodes in the model
* ``num_interactions``: number of interactions in the model
* ``cutoff``: cutoff radius in the model
* ``val_ratio``: validation set ratio
* ``forces_weight``: weight of the force loss. calculate forces if ``forces_weight > 0``
* ``normalization``: whether to use normalization
* ``atomwise_normalization``: whether to use atomwise normalization
* ``stop_patience``: patience for early stopping
* ``plateau_scheduler``: whether to use plateau scheduler
* ``random_seed``: random seed
* ``split_file``: path to the split file
* ``max_steps``: maximum number of steps
* ``max_epochs``: maximum number of epochs
* ``batch_size``: batch size
* ``initial_lr``: initial learning rate
* ``max_grad_norm``: maximum gradient norm
* ``optimizer_type``: optimizer type
* ``output_log``: output log file name
* ``log_interval``: log interval
* ``debug``: whether to use debug mode
* ``dist_timeout``: timeout for distributed training
* ``master_port``: master port for distributed training

There are more parameters for each model, please refer to the :doc:`api` reference for details.

.. note::
   Choose either ``max_steps`` or ``max_epochs`` to setup the training process. If both ``max_steps`` and ``max_epochs`` are set, the ``max_steps`` will be ignored.

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