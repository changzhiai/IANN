LAMMPS Interface
==============

This guide explains how to use IANN models as interatomic potentials in LAMMPS molecular dynamics simulations.

Exporting Models
-------------

To use an IANN model in LAMMPS, first export it to the correct format:

.. code-block:: python

   from iann.export import export_lammps
   from iann.models.painn import PainnModel
   import torch

   # Load your trained model
   state_dict = torch.load("model_output/best_model.pth")
   model = PainnModel(
       num_interactions=state_dict["num_layer"],
       hidden_state_size=state_dict["node_size"],
       cutoff=state_dict["cutoff"],
       compute_forces=True
   )
   model.load_state_dict(state_dict["model"])

   # Export for LAMMPS
   export_lammps(model, "model_lammps.pt")

LAMMPS Input Script
----------------

Here's a basic LAMMPS input script to use the exported model:

.. code-block:: lammps

   # Basic LAMMPS input script for IANN
   units metal
   atom_style atomic

   # Read structure
   read_data your_structure.data

   # Load ML potential
   pair_style mlip
   pair_coeff * * model_lammps.pt

   # Setup MD
   fix 1 all nve
   timestep 0.001
   thermo 100
   thermo_style custom step pe ke etotal temp press

   # Run MD
   run 10000

Key Components
------------

1. **Units and Style**
   * Use ``units metal`` for eV and Å
   * ``atom_style atomic`` for basic atomic systems

2. **Potential Setup**
   * ``pair_style mlip`` for ML interatomic potential
   * ``pair_coeff * * model_lammps.pt`` to load the model

3. **MD Settings**
   * Choose appropriate ensemble (NVE, NVT, NPT)
   * Set suitable timestep (typically 0.001 fs)
   * Configure output frequency

Advanced Usage
------------

1. **Different Ensembles**
   * NVT: Use ``fix nvt``
   * NPT: Use ``fix npt``
   * Custom: Use appropriate fix commands

2. **Output Options**
   * Energy components
   * Forces
   * Custom properties

3. **Performance Tuning**
   * Neighbor list settings
   * Communication settings
   * Parallelization options

Example: NVT Simulation
--------------------

.. code-block:: lammps

   # NVT simulation example
   units metal
   atom_style atomic
   read_data your_structure.data

   # Load potential
   pair_style mlip
   pair_coeff * * model_lammps.pt

   # Setup NVT
   velocity all create 300.0 12345
   fix 1 all nvt temp 300.0 300.0 0.1
   timestep 0.001
   thermo 100
   thermo_style custom step pe ke etotal temp press

   # Run
   run 10000

Troubleshooting
-------------

1. **Model Loading**
   * Verify model file exists
   * Check file permissions
   * Ensure correct format

2. **Performance Issues**
   * Adjust neighbor list settings
   * Check parallelization
   * Monitor memory usage

3. **Accuracy Concerns**
   * Verify cutoff radius
   * Check unit conversion
   * Validate energy/force scaling

For more details on LAMMPS usage and advanced features, refer to the `LAMMPS documentation <https://docs.lammps.org/>`_. 