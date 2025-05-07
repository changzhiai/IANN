API Reference
============

This section provides detailed documentation for the IANN package modules and classes.

Data
----

.. automodule:: iann.data.data
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: iann.data.data.AtomsData
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: iann.data.data.AseDataset
   :members:
   :undoc-members:
   :show-inheritance:

Models
------

.. .. automodule:: iann.models.painn
..    :members:
..    :undoc-members:
..    :show-inheritance:

.. autoclass:: iann.models.painn.PaiNN
   :members: __init__, forward

.. .. automodule:: iann.models.nequip
..    :members:
..    :undoc-members:
..    :show-inheritance:

.. autoclass:: iann.models.nequip.NequIP
   :members: __init__, forward

.. .. automodule:: iann.models.mace
..    :members:
..    :undoc-members:
..    :show-inheritance:

.. autoclass:: iann.models.mace.MACE
   :members: __init__, forward

.. .. automodule:: iann.models.equiformerV2
..    :members:
..    :undoc-members:
..    :show-inheritance:

.. autoclass:: iann.models.equiformerV2.EquiformerV2
   :members: __init__, forward


Calculators
--------

.. automodule:: iann.calculators.calculators.MLCalculator
   :members: __init__, forward

.. autoclass:: iann.calculators.calculators.EnsembleCalculator
   :members: __init__, forward

.. autoclass:: iann.calculators.calculators.AtomicEnsembleCalculator
   :members: __init__, forward

Trainer
------

.. automodule:: iann.trainer.trainer.Trainer
   :members: __init__, forward

Plugins
------

.. automodule:: iann.plugins.converter.LAMMPSModelWrapper
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
------------

The configuration system uses Dict or TOML file. Here are the available options:

.. code-block:: toml

   # Model parameters
   node_size = 128          # Size of hidden states
   num_interactions = 3     # Number of message passing layers
   cutoff = 4.0            # Cutoff radius for atomic interactions
   
   # Training parameters
   val_ratio = 0.1         # Validation set ratio
   output_dir = "output"   # Output directory
   dataset = "data.traj"   # Dataset path
   max_steps = 100000      # Maximum training steps
   batch_size = 32         # Batch size per GPU
   initial_lr = 0.0001     # Initial learning rate
   forces_weight = 0.9     # Weight for force loss
   
   # Logging and monitoring
   log_interval = 2000     # Logging interval
   normalization = true    # Enable energy normalization
   atomwise_normalization = true  # Enable per-atom normalization
   stop_patience = 50      # Early stopping patience
   random_seed = 42        # Random seed

Command Line Interface
--------------------

The training script can be run with various command line arguments:

.. code-block:: bash

   python test/painn/train.py --cfg config.toml [options]

Options:

* ``--cfg``: Path to configuration file (required)
* ``--resume``: Resume from checkpoint
* ``--debug``: Enable debug mode
* ``--no-cuda``: Disable CUDA
* ``--seed``: Set random seed

For more information about specific functions and classes, see their respective module documentation. 