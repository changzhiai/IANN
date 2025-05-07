Making Predictions
===============

This guide explains how to use trained IANN models for making predictions.

Loading a Trained Model
--------------------

To load a trained model:

.. code-block:: python

   from iann.models.painn import PainnModel
   import torch

   # Load model
   state_dict = torch.load("model_output/best_model.pth")
   model = PainnModel(
      num_interactions=state_dict["num_layer"],
      hidden_state_size=state_dict["node_size"],
      cutoff=state_dict["cutoff"],
      compute_forces=True
   )
   model.load_state_dict(state_dict["model"])

Using the ML Calculator
--------------------

IANN provides an ASE calculator interface for easy integration:

.. code-block:: python

   from iann.calculators.calculators import MLCalculator
   from ase.io import read

   # Create calculator
   calc = MLCalculator(model)

   atoms = read("test_structures.traj", ":")
   for atom in atoms:
      atom.calc = calc
      energy = atom.get_potential_energy()
      forces = atom.get_forces()
      print(f"Energy: {energy} eV")
      print(f"Forces: {forces} eV/Å")

Note: ``EnsembleCalculator`` and ``AtomicEnsembleCalculator`` are available to get uncertainty for each structure and each atom, respectively.

Batch Prediction
--------------

For efficient batch prediction:

.. code-block:: python

   from iann.data import AseDataset
   from torch.utils.data import DataLoader

   # Create dataset
   dataset = AseDataset("test_structures.traj")
   dataloader = DataLoader(dataset, batch_size=32)

   # Make predictions
   model.eval()
   for batch in dataloader:
      energy, forces = model(batch)
      # Process predictions...

Performance Tips
--------------

1. **Memory Management**

   * Use appropriate batch sizes
   * Clear GPU cache if needed

2. **Speed Optimization**

   * Enable CUDA if available
   * Use batch processing when possible
   * Consider model quantization for deployment

3. **Accuracy Considerations**

   * Check cutoff radius matches training
   * Verify atomic numbers are correct

Integration with Other Tools
-------------------------

IANN models can be used with various tools:

* ASE for structure manipulation
* LAMMPS for molecular dynamics (see :doc:`lammps`)
* Custom scripts for specific applications

For more details on the API and advanced usage, see the :doc:`api` reference. 