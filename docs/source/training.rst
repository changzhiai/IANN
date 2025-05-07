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

.. code-block:: python

   from iann.trainer.trainer import Trainer

   trainer = Trainer(
      model="painn",
      config={"device": "cpu", 
               'output_dir': 'output'},
      distributed=False
      )
   trainer.train("dataset.traj")

Available models for model:

* painn
* nequip
* mace
* equiformerV2

Available configurations for config:

.. code-block:: python

   config = {
      "max_steps": 50000,
      "node_size": 128,
      "num_interactions": 3,
      "cutoff": 4.0,
      "val_ratio": 0.1,
      "output_dir": "output",
      "dataset": "path/to/your/data.traj",
      "batch_size": 32,
      "initial_lr": 0.0001,
      "forces_weight": 0.9,
      "log_interval": 2000,
      "normalization": True,
      "stop_patience": 50,
      "random_seed": 666,
      "load_model": None,
      "device": "cpu",
   }


Multi-GPU Training
------------------

See the :doc:`parallelization` guide for details on distributed training.

Command Line Training
---------------------

Run the trainin in command line is another way. Put all configurations in a TOML file, and run it in a python command line.
.. code-block:: bash 

   python test/painn/train.py --cfg config.toml


Monitoring Training
-------------------

Training progress is logged to the specified output directory. You can monitor:

* Energy and force prediction errors
* Training and validation losses
* Model checkpoints

The best model is saved as ``best_model.pth`` in the output directory.

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