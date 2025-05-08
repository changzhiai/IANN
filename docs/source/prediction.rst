Predicting Guide
===============

This guide explains how to use trained IANN models for making predictions.


Using the ML Calculator
--------------------

The ``MLCalculator`` provides a convenient ASE calculator interface:

.. code-block:: python

   from iann.calculators.calculators import MLCalculator
   from ase.io import read

   # Create calculator with model path
   calc = MLCalculator("trained/best_model.pth")
   
   # Read structures
   images = read("test_structures.traj", ":")

   # Make predictions
   for atoms in images:
      atoms.calc = calc
      energy = atoms.get_potential_energy()
      forces = atoms.get_forces()
      print(f"Energy: {energy} eV")
      print(f"Forces: {forces} eV/Å")

The calculator automatically:

* Determines the model type from the saved state dict
* Use the model architecture
* Do prediction

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