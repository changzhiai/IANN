Engine Models
==============

This guide covers the backend models available in IANN, including their architectures, features, and use cases.

Overview
--------

IANN provides several state-of-the-art foundation models for interatomic potentials. All of them are graph-based equivariant neural networks:

- `FastPot <https://github.com/changzhiai/IANN/blob/master/iann/models/fastpot.py>`_ (Fast Potential with high-order tensor features and equivariant message passing)
- `PaiNN <https://arxiv.org/abs/2102.03150>`_ (Polarizable atom interaction Neural Network)
- `NequIP <https://doi.org/10.1038/s41467-022-29939-5>`_ (Neural equivariant Interatomic Potentials)
- `MACE <https://arxiv.org/abs/2206.07697>`_ (Message-passing Atomic Cluster Expansion)
- `EquiformerV2 <https://arxiv.org/abs/2306.12059>`_ (Equivariant Transformer V2)

Each model has its own strengths and is suitable for different applications.

.. note::
   NVIDIA Python library ``cuEquivariance`` is an optional dependency for the ``NequIP`` and ``MACE`` models. If you want to use the optimized operations, you need to install it, and the models will automatically use it. Please check the official website `cuEquivariance <https://docs.nvidia.com/cuda/cuequivariance/>`_ for more details.

FastPot
-------

FastPot is a high-performance neural network model that combines the power of high-order tensor features and equivariant message passing for fast and accurate potential energy surface prediction.

Key features:

* Very high computational efficiency
* Good balance of accuracy and speed
* Suitable for general-purpose applications

Example usage:

.. code-block:: python

   from iann.models.fastpot import FastPot
   
   model = FastPot(
       num_layers=3,
       num_channels=128,
       cutoff=5.5,
       compute_forces=True
   )

There are more adjustable parameters for FastPot model to setup the training process, please check the source code :class:`iann.models.fastpot.FastPot` for more details (all adjustable parameters are passed as `kwargs.get` in the model class).

PaiNN
-----

PaiNN is a message-passing neural network that considers features including:

* Scalar: atomic number, distance
* Vector: coordinate difference

Key features:

* High computational efficiency
* Good balance of accuracy and speed
* Suitable for general-purpose applications

Example usage:

.. code-block:: python

   from iann.models.painn import PaiNN
   
   model = PaiNN(
       num_layers=3,
       num_channels=128,
       cutoff=5.5,
       compute_forces=True
   )

There are more adjustable parameters for PaiNN model to setup the training process, please check the source code :class:`iann.models.painn.PaiNN` for more details (all adjustable parameters are passed as `kwargs.get` in the model class).

NequIP
------

NequIP is an equivariant neural network that considers features including:

* Scalar: atomic number, distance
* Vector: coordinate difference
* Higher-order tensor: high order rotation


Key features:

* Excellent accuracy
* Uses spherical harmonics symmetries
* Good for high-precision applications

Example usage:

.. code-block:: python

   from iann.models.nequip import NequIP
   
   model = NequIP(
       num_layers=3,
       num_channels=64,
       cutoff=5.5,
       compute_forces=True
   )

There are more adjustable parameters for NequIP model to setup the training process, please check the source code :class:`iann.models.nequip.NequIP` for more details (all adjustable parameters are passed as `kwargs.get` in the model class).

MACE
----

MACE combines message-passing architecture and multi-body expansion, which considers features including:

* Scalar: atomic number, distance
* Vector: coordinate difference
* Higher-order tensor: high order rotation
* multi-body expansion

Key features:

* Fast training and inference
* Good scaling properties
* Suitable for large-scale applications

Example usage:

.. code-block:: python

   from iann.models.mace import MACE
   
   model = MACE(
       num_layers=3,
       num_channels=64,
       cutoff=5.5,
       compute_forces=True
   )

There are more adjustable parameters for MACE model to setup the training process, please check the source code :class:`iann.models.mace.MACE` for more details (all adjustable parameters are passed as `kwargs.get` in the model class).

EquiformerV2
-----------

EquiformerV2 is a transformer-based model that:

* Uses attention mechanisms
* Preserves physical symmetries


Key features:

* State-of-the-art accuracy
* Good for complex systems

Example usage:

.. code-block:: python

   from iann.models.equiformerV2 import EquiformerV2
   
   model = EquiformerV2(
       num_layers=3,
       num_channels=32,
       cutoff=5.5,
       compute_forces=True
   )

There are more adjustable parameters for EquiformerV2 model to setup the training process, please check the source code :class:`iann.models.equiformerV2.EquiformerV2` for more details (all adjustable parameters are passed as `kwargs.get` in the model class).

Model Selection
-------------

When choosing a model, consider:

1. **Accuracy Requirements**

   * NequIP, MACE and EquiformerV2 for highest accuracy
   * FastPot, PaiNN for balanced performance

2. **Computational Resources**

   * FastPot and PaiNN for fast training/inference
   * NequIP, MACE and EquiformerV2 for most complex systems

3. **System Size**

   * FastPot, PaiNN for super large systems
   * NequIP, MACE for large systems
   * EquiformerV2 is not the best choice for small systems

.. note::

   The description may be not reliable and the reliable data will be provided in future.

For detailed API documentation of each model, see the :doc:`api` reference. 