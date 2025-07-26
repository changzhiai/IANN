Engine Models
==============

This guide covers the backend models available in IANN, including their architectures, features, and use cases.

Overview
--------

IANN provides several state-of-the-art foundation models for interatomic potentials. All of them are graph-based equivariant neural networks:

- `PaiNN <https://arxiv.org/abs/2102.03150>`_ (Polarizable atom interaction Neural Network)
- `NequIP <https://doi.org/10.1038/s41467-022-29939-5>`_ (Neural equivariant Interatomic Potentials)
- `MACE <https://arxiv.org/abs/2206.07697>`_ (Message-passing Atomic Cluster Expansion)
- `EquiformerV2 <https://arxiv.org/abs/2306.12059>`_ (Equivariant Transformer V2)

Each model has its own strengths and is suitable for different applications.

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
       num_channels=64,
       cutoff=5.5,
       compute_forces=True
   )

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

Model Selection
-------------

When choosing a model, consider:

1. **Accuracy Requirements**

   * MACE and EquiformerV2 for highest accuracy
   * PaiNN and MACE for balanced performance

2. **Computational Resources**

   * PaiNN for fastest training/inference
   * EquiformerV2 for most complex systems

3. **System Size**

   * PaiNN and MACE for large systems
   * NequIP for smaller, high-precision systems
   * EquiformerV2 is not the best choice for small systems

.. note::

   The description may be not reliable and the reliable data will be provided in future.

For detailed API documentation of each model, see the :doc:`api` reference. 