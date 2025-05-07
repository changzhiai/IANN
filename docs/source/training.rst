Training Guide
=============

This guide covers how to train IANN models for energy and force prediction.

Preparing Your Dataset
--------------------

IANN works with ASE database (.db) or trajectory (.traj) files. Your data should contain:

* Atomic structures (positions, atomic numbers)
* Energy labels
* Force labels (optional, but recommended)

Configuration
------------

Training is configured using a TOML file. Here's an example configuration:

.. code-block:: toml

   node_size = 128
   num_interactions = 3
   cutoff = 4.0
   val_ratio = 0.1
   output_dir = "model_output"
   dataset = "path/to/your/data.traj"
   max_steps = 100000
   batch_size = 32
   initial_lr = 0.0001
   forces_weight = 0.9
   log_interval = 2000
   normalization = true
   atomwise_normalization = true
   stop_patience = 50
   random_seed = 42

Key parameters:
* ``node_size``: Size of hidden states
* ``num_interactions``: Number of message passing layers
* ``cutoff``: Cutoff radius for atomic interactions
* ``forces_weight``: Weight for force loss in total loss
* ``normalization``: Whether to normalize energies
* ``atomwise_normalization``: Whether to normalize per atom

Running Training
--------------

Single-GPU Training
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python test/painn/train.py --cfg config.toml

Multi-GPU Training
~~~~~~~~~~~~~~~~

See the :doc:`parallelization` guide for details on distributed training.

Monitoring Training
----------------

Training progress is logged to the specified output directory. You can monitor:

* Energy and force prediction errors
* Training and validation losses
* Model checkpoints

The best model is saved as ``best_model.pth`` in the output directory.

Training Tips
-----------

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