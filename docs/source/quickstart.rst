Quickstart Guide
===============

This guide will help you get started with IANN quickly. We'll cover the basic usage of the package for training and prediction.

Basic Example
------------

Here's a simple example that demonstrates how to use IANN:

.. code-block:: python

   from iann.models.painn import PainnModel
   from iann.data import AseDataset
   from iann.utils import MLCalculator
   from ase.io import read

   # Create a PaiNN model
   model = PainnModel(
       num_interactions=3,
       hidden_state_size=128,
       cutoff=4.0,
       compute_forces=True
   )

   # Load your dataset
   dataset = AseDataset("path/to/your/data.traj")

   # Create a calculator for predictions
   calc = MLCalculator(model)

   # Make predictions
   atoms = read("test_structures.traj", ":")
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

   # Run the quickstart example
   python examples/quickstart.py

This script demonstrates:
* Loading a dataset
* Creating and training a PaiNN model
* Evaluating the model
* Using the model for predictions

Next Steps
----------

After running the quickstart example, you might want to:

1. Check out the :doc:`training` guide for detailed training instructions
2. Learn about :doc:`prediction` for making predictions with trained models
3. Explore :doc:`parallelization` for multi-GPU training
4. Read about :doc:`lammps` for using IANN with LAMMPS

For more examples and tutorials, visit the `examples/` directory in the IANN repository. 