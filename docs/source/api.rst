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

.. cpp:class:: PairIANN
   :members:

.. cpp:class:: PairIANNMultiGPU
   :members:

.. cpp:class:: ComputeIANNVariance
   :members:


For more information about specific functions and classes, see their respective module in source code. 