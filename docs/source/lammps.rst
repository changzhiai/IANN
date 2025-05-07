LAMMPS Interface
==============

This guide explains how to use IANN models as interatomic potentials in LAMMPS molecular dynamics simulations.

Exporting Models
-------------

To use an IANN model in LAMMPS, first export it to the correct format:

.. code-block:: python

    from iann.plugins.converter import convert_model_for_lammps

    convert_model_for_lammps(model_path='best_model.pth', 
                            model_type='painn', 
                            output_path='output_model.pth')

LAMMPS Input Script
----------------

Here's a basic LAMMPS input script to use the exported model:

.. code-block:: lammps

    # LAMMPS input script example
    units metal
    atom_style atomic
    boundary p p p

    read_data initial.data

    # Define the IANN pair style
    pair_style iann painn output_model.pt 5.5
    pair_coeff * *

    mass 1 1.0079999997406976 # H
    mass 2 195.08399994981576 # Pt

    neighbor 0.5 bin
    neigh_modify every 1 delay 0 check yes

    # Thermodynamic settings
    thermo 10

    # Initial minimization to relax the system before dynamics
    minimize 1.0e-4 1.0e-6 100 1000

    # Run your simulation
    timestep 0.001
    fix 1 all nvt temp 300.0 300.0 0.1
    dump 1 all custom 10 dump.xyz id type x y z

    run 5000

Key Components
------------

1. **Units and Style**

   * Use ``units metal`` for eV and Å
   * ``atom_style atomic`` for basic atomic systems

2. **Potential Setup**

   * ``pair_style iann painn output_model.pt 5.5`` for loading model

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
   pair_style iann painn output_model.pt 5.5
   pair_coeff * *

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