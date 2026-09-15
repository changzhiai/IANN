Parallelization Guide
===================

Serial Training
---------------

Before going to parallelization, we first take a look at the single GPU/CPU training.

Here is the training script ``train.py``:

.. code-block:: python

   from iann.trainer import Trainer

   # Define the trainer
   trainer = Trainer(
      model="painn",
      config={"device": "cuda", 
              'output_dir': 'output',
              'output_log': 'output.log',
              'output_model': 'model.pt'},
      distributed=False,
      )

   # Train the model
   trainer.train("dataset.traj")

To run **training on a local machine** in serial, you can run the script on command line:

.. code-block:: bash

   # Run on a local machine
   python train.py

or you can submit it to a `SLURM <https://slurm.schedmd.com/>`_ workload manager.

To run **training on a single GPU**:

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

To run **training on a single CPU**:

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


**Multi-GPU Training**: submit to multiple GPUs and multiple nodes (in SLURM Workload Manager)

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
   
   export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
   export NNODES=$SLURM_NNODES

   srun -N $NNODES -n $((NNODES*GPUS_PER_NODE)) python train.py

**Multi-CPU Training**: submit to multiple CPUs and multiple nodes (in SLURM Workload Manager)

.. code-block:: bash

   #!/bin/bash
   #SBATCH -N 2                   # Number of nodes
   #SBATCH -C cpu                 # Use CPU nodes
   #SBATCH -q debug               # Use regular/debug queue
   #SBATCH -t 00:30:00            # Time limit
   #SBATCH -A m2997               # Your account
   #SBATCH --ntasks-per-node=1    # Number of tasks per node
   #SBATCH --cpus-per-task=128    # Number of CPUs per task

   module load your_modules

   export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
   export NNODES=$SLURM_NNODES

   srun -N $NNODES -n $((NNODES*GPUS_PER_NODE)) python train.py



To directly run **training on multiple GPUs/CPUs on a local machine**:

.. code-block:: bash

   # Run on N GPUs/CPUs
   python -m torch.distributed.launch --nproc_per_node=N train.py

Or use the following command by ``torchrun``:

.. code-block:: bash

   # Run on N GPUs/CPUs
   torchrun --nproc_per_node=N train.py


.. note::
   nproc_per_node defines the number of local CPU or GPU workers. Setup ``device="cpu"`` or ``device="cuda"`` to make device selection in your config.toml.


Examples of parallel training on NERSC, S3DF and Carbon
-------------------------------------------------------

Here is an example of how to run **multi-GPU training on NERSC**:

.. code-block:: bash

   #!/bin/bash
   #SBATCH -N 2                   # Number of nodes
   #SBATCH -C gpu                 # Use GPU nodes
   #SBATCH -q regular             # Use regular/debug queue
   #SBATCH -t 00:30:00            # Time limit
   #SBATCH -A mxxxx               # Your account
   #SBATCH --gpus-per-node=4      # GPUs per node
   #SBATCH --ntasks-per-node=4    # Number of tasks per node
   #SBATCH --cpus-per-task=1      # Number of CPUs per task

   # Load the environments, such as:
   export PYTHONPATH=/path/to/IANN/:$PYTHONPATH
   module purge
   module load PrgEnv-nvidia; module load openmpi

   # Vender bugs fixed on nersc for multiple nodes
   export FI_CXI_RDZV_GET_MIN=0 # vender bugs fixed on nersc for multiple nodes
   export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216 # vender bugs fixed on nersc

   # GPUs per node and number of nodes
   export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
   export NNODES=$SLURM_NNODES

   # Run the training script on multiple GPUs/CPUs
   srun -N $NNODES -n $((NNODES*GPUS_PER_NODE)) python train.py

.. -A m2997
.. export PYTHONPATH=/pscratch/sd/c/changzhi/softwares/IANN_v3/IANN/:$PYTHONPATH

Here is an example of how to run **multi-GPU training on S3DF**:

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

   # Load the environments, such as:
   conda activate venv_xxxx
   export PYTHONPATH=/path/to/IANN:$PYTHONPATH

   # GPUs per node and number of nodes
   export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
   export NNODES=$SLURM_NNODES

   # Run the training script on multiple GPUs/CPUs
   srun -N $NNODES -n $((NNODES*GPUS_PER_NODE)) python train.py

.. conda activate /sdf/home/c/changzhi/softwares/anoconda3/envs/painn
.. export PYTHONPATH=/sdf/home/c/changzhi/changzhi/softwares/IANN_v3/IANN:$PYTHONPATH

.. note::
   The ``srun`` command is used to run the training script on multiple GPUs/CPUs. It is a wrapper around the ``mpirun`` command for multiple GPUs/CPUs parallelization.
   For multi-GPUs/multi-CPUs mode, `Trainer.train()` will call `mp.spawn()` to launch `world_size` workers using the `process_function`.
   The parallelization parameters would be automatically obtained from the SLURM environment variables.


Here is an example of how to run **multi-GPU training on Carbon**, which uses
PBS rather than SLURM, so the ranks are launched with ``mpirun``:

.. code-block:: bash

   #!/bin/bash

   #PBS -l nodes=1:ppn=4:gpus=2
   #PBS -l walltime=5:00:00
   #PBS -N train
   #PBS -A cnmxxxx
   #PBS -o job.out
   #PBS -e job.err

   # Load the environments, such as:
   cd $PBS_O_WORKDIR
   source ~/miniconda/etc/profile.d/conda.sh
   conda activate base
   module load openmpi
   export PYTHONPATH=/path/to/IANN:$PYTHONPATH

   # GPUs per node and number of nodes
   GPUS_PER_NODE=2
   export MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")   # all ranks rendezvous here
   export MASTER_PORT=12356

   # environment forwarded to the remote ranks (conda/python, libs, rendezvous)
   FWD="-x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x MASTER_ADDR -x MASTER_PORT"

   mpirun --map-by ppr:${GPUS_PER_NODE}:node -machinefile "$PBS_NODEFILE" $FWD python train.py

.. -A cnm84157
.. export PYTHONPATH=/home/changzhi/softwares/IANN_dev/fix-mace/IANN:$PYTHONPATH

Three things differ from the SLURM examples, and all three matter:

* **The rank comes from MPI, not the scheduler.** With no ``SLURM_PROCID`` to
  read, the trainer falls back to ``OMPI_COMM_WORLD_RANK`` /
  ``OMPI_COMM_WORLD_SIZE`` (and the MPICH/Intel and MVAPICH2 equivalents). The
  training script itself is unchanged — it still only needs
  ``distributed=True``.
* ``--map-by ppr:N:node`` **replaces** ``-np``. It places *N* ranks per node for
  every node in ``$PBS_NODEFILE``, so one rank lands on each GPU without having
  to compute the total rank count by hand.
* **The environment must be forwarded explicitly.** ``mpirun`` does not carry
  your shell environment to remote nodes, so ``PATH``, ``LD_LIBRARY_PATH`` and
  ``PYTHONPATH`` are passed with ``-x`` — otherwise the remote ranks start with
  the system Python and fail to import ``iann``.

.. note::
   ``MASTER_ADDR`` is exported here for clarity, but it is optional on PBS: if it
   is unset the trainer reads the first host from ``$PBS_NODEFILE`` itself,
   falling back to ``localhost`` on a single node. What is **not** optional is
   forwarding it with ``-x`` once you set it — on multiple nodes every rank must
   rendezvous on the same host, and a rank that does not receive the variable
   will pick its own and hang.


Parallelization Configuration
-----------

When using DDP, keep in mind that these ``config`` parameters are vital, which are automatically obtained from the SLURM environment variables when using SLURM Workload Manager:

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