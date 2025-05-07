Making Predictions
===============

This guide explains how to use trained IANN models for making predictions.

Loading a Trained Model
--------------------

To load a trained model:

.. code-block:: python

   from iann.models.painn import PainnModel
   import torch

   # Load model state dict
   state_dict = torch.load("model_output/best_model.pth")
   
   # Create model with same architecture
   model = PainnModel(
       num_interactions=state_dict["num_layer"],
       hidden_state_size=state_dict["node_size"],
       cutoff=state_dict["cutoff"],
       compute_forces=True
   )
   
   # Load weights
   model.load_state_dict(state_dict["model"])

Using the ML Calculator
--------------------

IANN provides an ASE calculator interface for easy integration:

.. code-block:: python

   from iann.utils import MLCalculator
   from ase.io import read

   # Create calculator
   calc = MLCalculator(model)

   # Read structures
   atoms = read("test_structures.traj", ":")
   
   # Make predictions
   for atom in atoms:
       atom.calc = calc
       energy = atom.get_potential_energy()
       forces = atom.get_forces()
       print(f"Energy: {energy} eV")
       print(f"Forces: {forces} eV/Å")

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
   with torch.no_grad():
       for batch in dataloader:
           energy, forces = model(batch)
           # Process predictions...

Performance Tips
--------------

1. **Memory Management**
   * Use appropriate batch sizes
   * Clear GPU cache if needed
   * Use ``torch.no_grad()`` for inference

2. **Speed Optimization**
   * Enable CUDA if available
   * Use batch processing when possible
   * Consider model quantization for deployment

3. **Accuracy Considerations**
   * Ensure input data is properly normalized
   * Check cutoff radius matches training
   * Verify atomic numbers are correct

Integration with Other Tools
-------------------------

IANN models can be used with various tools:

* ASE for structure manipulation
* LAMMPS for molecular dynamics (see :doc:`lammps`)
* Custom scripts for specific applications

For more details on the API and advanced usage, see the :doc:`api` reference. 