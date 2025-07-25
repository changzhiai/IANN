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

.. autoclass:: iann.models.painn.PaiNN
   :members: forward


.. autoclass:: iann.models.nequip.NequIP
   :members: forward


.. autoclass:: iann.models.mace.MACE
   :members: forward


.. autoclass:: iann.models.equiformerV2.EquiformerV2
   :members: forward

Trainer
------

.. autoclass:: iann.trainer.Trainer
   :members: __init__, forward


Calculators
--------

.. autoclass:: iann.calculators.MLCalculator
   :members: forward

.. autoclass:: iann.calculators.EnsembleCalculator
   :members: forward

.. autoclass:: iann.calculators.AtomicEnsembleCalculator
   :members: forward


Plugins
------

.. autoclass:: iann.plugins.converter.LAMMPSModelWrapper
   :members: forward

.. autoclass:: iann.plugins.converter.EnsembleLAMMPSModelWrapper
   :members: forward

.. autofunction:: iann.plugins.converter.convert_model_for_lammps

.. autofunction:: iann.plugins.converter.convert_models_for_lammps


For more information about specific functions and classes, see their respective module in source code. 