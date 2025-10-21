API Reference
============

This section provides detailed documentation and source code links for the IANN package modules, classes, and functions.

Data
----

.. autoclass:: iann.data.AtomsData
   :members: forward

.. autoclass:: iann.data.AseDataset
   :members: forward

Models
------
.. autoclass:: iann.models.fastpot.FastPot
   :members: __init__, forward

.. autoclass:: iann.models.painn.PaiNN
   :members: __init__, forward


.. autoclass:: iann.models.nequip.NequIP
   :members: __init__, forward


.. autoclass:: iann.models.mace.MACE
   :members: __init__, forward


.. autoclass:: iann.models.equiformerV2.EquiformerV2
   :members: __init__, forward

Trainer
------

.. autoclass:: iann.trainer.Trainer
   :members: __init__, forward


Calculators
----------

.. autoclass:: iann.calculators.MLCalculator
   :members: __init__, calculate

.. autoclass:: iann.calculators.EnsembleCalculator
   :members: __init__, calculate, get_ensemble

.. autoclass:: iann.calculators.AtomicEnsembleCalculator
   :members: __init__, calculate, get_ensemble


Plugins
------

.. autoclass:: iann.plugins.converter.LAMMPSModelWrapper
   :members: __init__, forward

.. autoclass:: iann.plugins.converter.EnsembleLAMMPSModelWrapper
   :members: __init__, forward

.. autofunction:: iann.plugins.converter.convert_model_for_lammps

.. autofunction:: iann.plugins.converter.convert_models_for_lammps

C++ LAMMPS Plugins
------------------

IANN Pair Style (Single GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:class:: PairIANN

   LAMMPS pair style for using trained IANN potentials in molecular dynamics simulations.

   **LAMMPS Command:** ``pair_style iann model_type model_path cutoff``

   **Parameters:**

   - ``model_type``: Type of ML model (fastpot, painn, nequip, mace, equiformer2)
   
   - ``model_path``: Path to the exported TorchScript model file
   
   - ``cutoff``: Interaction cutoff distance in Å

IANN Pair Style (Multiple GPUs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:class:: PairIANNMultiGPU

   Multi-GPU version of IANN pair style with automatic GPU detection and utilization.

   **LAMMPS Command:** ``pair_style iann/multi_gpu model_type model_path cutoff``

   **Parameters:** Same as single GPU version

IANN Variance Compute Style
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:class:: ComputeIANNVariance

   Compute style for accessing ensemble variance statistics from IANN potentials.

   **LAMMPS Command:** ``compute variance all iann/variance``

   **Output Components:**

   - ``c_variance[1]``: Energy variance
   
   - ``c_variance[2]``: Force variance  
   
   - ``c_variance[3]``: Maximum energy variance
   
   - ``c_variance[4]``: Maximum force variance

   **Usage Example:**
   ::

      # Set up compute for variance
      compute variance all iann/variance
      
      # Use in thermo output
      thermo_style custom step pe ke etotal press c_variance[1] c_variance[2] c_variance[3] c_variance[4]
      
      thermo_modify colname c_variance[1] energy_variance
      thermo_modify colname c_variance[2] force_variance
      thermo_modify colname c_variance[3] max_energy_variance
      thermo_modify colname c_variance[4] max_force_variance


For more information about specific functions and classes, see their respective module in source code. 