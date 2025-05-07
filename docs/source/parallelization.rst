Parallelization Guide
===================

This guide covers how to use IANN with distributed training for better performance.

Distributed Data Parallel (DDP)
----------------------------

IANN supports distributed training using PyTorch's Distributed Data Parallel (DDP). This allows training on multiple GPUs efficiently.

Basic Usage
----------
Multi-GPU Training: submit to multiple GPUs (in SLURM Workload Manager)
.. code-block:: bash
   # Run on multiple GPUs and multiple nodes
   #!/bin/bash
   #SBATCH -N 2                   # Number of nodes
   #SBATCH -C gpu                 # Use GPU nodes
   #SBATCH -q debug
   #SBATCH -t 00:30:00 
   #SBATCH --gpus-per-node=4      # Number of GPUs per node
   module load pytorch
   srun python run.py

Multi-CPU Training: submit to multiple CPUs (in SLURM Workload Manager)
.. code-block:: bash
   # Run on multiple CPUs and multiple nodes
   #!/bin/bash
   #SBATCH -N 2                   # Number of nodes
   #SBATCH -C cpu                 # Use CPU nodes
   #SBATCH -q debug
   #SBATCH -t 00:30:00
   #SBATCH --ntasks-per-node=128  # Number of CPU cores per node
   module load pytorch
   srun python run.py

Command Line Interface
----------

To run training on multiple GPUs:

.. code-block:: bash

   # Run on N GPUs
   python -m torch.distributed.launch --nproc_per_node=N test/painn/train.py --cfg config.toml

For example, to use 4 GPUs:

.. code-block:: bash

   python -m torch.distributed.launch --nproc_per_node=4 test/painn/train.py --cfg config.toml

Configuration
-----------

When using DDP, consider these configuration parameters:

.. code-block:: toml

   # DDP-specific settings
   batch_size = 32  # This is per GPU
   num_workers = 4  # DataLoader workers per GPU
   ddp_backend = "nccl"  # Use NCCL for GPU training
   ddp_timeout = 1800  # Timeout in seconds

Performance Optimization
--------------------

1. **Batch Size**
   * Set batch_size per GPU
   * Total batch size = batch_size * num_gpus
   * Adjust based on GPU memory

2. **Data Loading**
   * Use multiple workers per GPU
   * Enable pin_memory for faster data transfer
   * Consider using persistent workers

3. **Communication**
   * Use NCCL backend for GPU training
   * Set appropriate timeout values
   * Monitor GPU utilization

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