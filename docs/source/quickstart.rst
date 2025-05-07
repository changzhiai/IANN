Quickstart
===============

This guide will help you get started with IANN quickly. We'll cover the basic usage of the package for training and prediction.

Basic Example
------------

Here's a simple example that demonstrates how to use IANN:

.. code-block:: python

    from iann.models.painn import PainnModel
    from iann.trainer.trainer import Trainer
    import torch
    from iann.calculators.calculators import MLCalculator
    from ase.io import read

    # Train a model
    trainer = Trainer(
        model="painn",
        config={"device": "cpu", 
                'output_dir': 'output'},
        distributed=False
        )
    trainer.train("dataset.traj")

    # Load model
    state_dict = torch.load("best_model.pth")
    model = PainnModel(
        num_interactions=state_dict["num_layer"],
        hidden_state_size=state_dict["node_size"],
        cutoff=state_dict["cutoff"],
        compute_forces=True
    )
    model.load_state_dict(state_dict["model"])

    # Create calculator
    calc = MLCalculator(model)

    # Predict
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

   python examples/quickstart.py

This script demonstrates:
* Loading a dataset
* Creating and training a model
* Using the model for predictions

Next Steps
----------

After running the quickstart example, you might want to:

1. Check out the :doc:`training` guide for detailed training instructions
2. Learn about :doc:`prediction` for making predictions with trained models
3. Explore :doc:`parallelization` for multi-GPU training
4. Read about :doc:`lammps` for using IANN with LAMMPS

For more examples and tutorials, visit the `examples/` directory in the IANN repository. 