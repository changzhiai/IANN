Quickstart
===============

This guide will help you get started with IANN quickly. We'll cover the basic usage of the package for training and prediction.

Basic Example
------------

Here's a simple example that demonstrates how to use IANN:

.. code-block:: python

    from iann.trainer import Trainer
    from iann.calculators import MLCalculator
    from ase.io import read

    # Train a model
    trainer = Trainer(
        model="painn",
        config={"device": "cpu", 
                'output_dir': 'output',
                'output_log': 'output.log',
                'output_model': 'model.pt'},
        distributed=False
        )
    
    # Prepare dataset and train model
    trainer.train("dataset.traj")

    # Create calculator with trained model
    calc = MLCalculator("output/model.pt")

    # Read structures
    atoms = read("test_structures.traj", ":")

    # Make predictions
    for atom in atoms:
        atom.calc = calc
        energy = atom.get_potential_energy()
        forces = atom.get_forces()
        print(f"Energy: {energy} eV")
        print(f"Forces: {forces} eV/Å")

Running the Example Script
-------------------------

IANN comes with example scripts to help you get started:

.. code-block:: bash

   # Run on a local machine
   python examples/quickstart.py

This script demonstrates:

* Loading a dataset
* Creating and training a model
* Using the model for predictions

Next Steps
----------

After running the quickstart example, you might want to:

1. Check out the :doc:`training` for detailed training instructions
2. Learn about :doc:`prediction` for making predictions with trained models
3. Explore :doc:`parallelization` for multi-GPU training
4. Read about :doc:`lammps` for using IANN with LAMMPS

For more examples and tutorials, visit the `examples/` directory in the IANN repository. 