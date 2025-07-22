Parallelization Guide
===================

Serial Training
---------------

Before going to parallelization, we first take a look at the single GPU/CPU training.

Here is the training script ``train.py``:

.. code-block:: python

   from iann.trainer.trainer import Trainer

   trainer = Trainer(
      model="painn",
      config={"device": "gpu", 
               'output_dir': 'output'},
      distributed=False
      )
   trainer.train("dataset.traj")

You can run the script on command line:

.. code-block:: bash

   python train.py

or you can submit it to a `SLURM <https://slurm.schedmd.com/>`_ workload manager.

To run training on a single GPU:
.. code-block:: bash

   #!/bin/bash
   #SBATCH -N 1                   # Number of nodes
   #SBATCH -C gpu                 # Use GPU nodes
   #SBATCH -q debug               # Use regular/debug queue
   #SBATCH -t 00:30:00            # Time limit
   #SBATCH -A m2997               # Your account
   #SBATCH --gpus-per-node=1      # GPUs per node
   #SBATCH --ntasks-per-node=1    # Number of tasks per node
   #SBATCH --cpus-per-task=1      # Number of CPU cores per task

   module load your_modules
   
   python train.py

To run training on single CPU:

.. code-block:: bash

   #!/bin/bash
   #SBATCH -N 1                   # Number of nodes
   #SBATCH -C cpu                 # Use CPU nodes
   #SBATCH -q debug               # Use regular/debug queue
   #SBATCH -t 00:30:00            # Time limit
   #SBATCH -A m2997               # Your account
   #SBATCH --ntasks-per-node=1    # Number of tasks per node
   #SBATCH --cpus-per-task=128      # Number of CPU cores per task

   module load your_modules
   
   python train.py

Parallel Training
-----------------

This guide covers how to use IANN with distributed training to parallelize the training process for speeding up the training. IANN supports distributed training using PyTorch's Distributed Data Parallel (DDP). This allows training on multiple GPUs efficiently.


Multi-GPU Training: submit to multiple GPUs and multiple nodes (in SLURM Workload Manager)

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

   module load your_modules
   
   srun python train.py

Multi-CPU Training: submit to multiple CPUs and multiple nodes (in SLURM Workload Manager)

.. code-block:: bash

   #!/bin/bash
   #SBATCH -N 2                   # Number of nodes
   #SBATCH -C cpu                 # Use CPU nodes
   #SBATCH -q debug               # Use regular/debug queue
   #SBATCH -t 00:30:00            # Time limit
   #SBATCH -A m2997               # Your account
   #SBATCH --ntasks-per-node=1    # Number of tasks per node
   #SBATCH --cpus-per-task=128      # Number of CPUs per task

   module load your_modules

   srun python train.py



To directly run training on multiple GPUs/CPUs in a single node:

.. code-block:: bash

   # Run on N GPUs/CPUs
   python -m torch.distributed.launch --nproc_per_node=N train.py

.. note::
   nproc_per_node defines the number of local CPU or GPU workers. Setup ``device="cpu"`` or ``device="cuda"`` to make device selection in your config.toml.


Examples of parallel training on NERSC and S3DF
----------------------------------------------

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

   export PYTHONPATH=/pscratch/sd/c/changzhi/softwares/IANN_v2/IANN/:$PYTHONPATH
   module purge
   module load PrgEnv-nvidia; module load openmpi

   export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
   export NNODES=$SLURM_NNODES
   export FI_CXI_RDZV_GET_MIN=0 # vender bugs fixed on nersc for multiple nodes
   export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216 # vender bugs fixed on nersc

   srun -N $NNODES -n $((NNODES*GPUS_PER_NODE)) \
      python train.py


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

   conda activate /sdf/home/c/changzhi/softwares/anoconda3/envs/painn
   export PYTHONPATH=/sdf/home/c/changzhi/changzhi/softwares/IANN_v2/IANN:$PYTHONPATH

   export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
   export NNODES=$SLURM_NNODES

   srun -N $NNODES -n $((NNODES*GPUS_PER_NODE)) \
      python train.py


Configuration
-----------

When using DDP, keep in mind that these configuration parameters are vital, which are automatically obtained from the SLURM environment variables when using SLURM Workload Manager:

.. code-block:: toml

   # DDP-specific settings
   batch_size = 32  # This is per GPU
   dist_timeout = 600  # Timeout in seconds
   master_port = 12356 # Master port for distributed training

Performance Optimization
------------------------

1. **Batch Size**

   * Set batch_size per GPU by ``batch_size = 32``
   * Total batch size = batch_size x num_gpus
   * Adjust based on GPU memory

2. **Data Loading**

   * Use multiple workers per GPU
   * Enable pin_memory for faster data transfer
   * Consider using persistent workers

3. **Communication**

   * Use ``dist_timeout`` to set the timeout for communication
   * Use ``master_port`` to set the master port for distributed training
   * If port conflicts, you can change the port by ``master_port = 12356``

Common Issues
-----------

1. **Gradient Strides Warning**

   * You may see a warning about gradient strides not matching bucket view strides
   * This is not an error and typically doesn't affect training
   * Can be safely ignored in most cases

2. **Memory Issues**

   * Reduce batch size if OOM errors occur
   * Monitor GPU memory usage
   * Consider gradient checkpointing for large models

3. **Communication Errors**

   * Check network connectivity between nodes
   * Verify NCCL installation
   * Adjust timeout values if needed

Best Practices
------------

1. **Scaling**

   * Start with a small number of GPUs
   * Monitor scaling efficiency
   * Adjust batch size and learning rate accordingly

2. **Monitoring**

   * Use tools like nvidia-smi to monitor GPU usage
   * Check communication overhead
   * Profile training for bottlenecks

3. **Debugging**

   * Run with a single GPU first
   * Enable debug logging if needed
   * Check for synchronization issues

For more advanced usage and troubleshooting, see the :doc:`api` reference. 