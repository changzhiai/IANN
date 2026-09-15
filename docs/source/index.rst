.. IANN documentation master file, created by
   sphinx-quickstart on Tue May  6 16:35:03 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to IANN's documentation!
================================

IANN (InterAtomic Neural Network Framework) is an equivariant interatomic neural network potential framework package for materials science and computational chemistry. It implements state-of-the-art graph neural network models for periodic and non-periodic systems, including PaiNN, NequIP, Allegro, MACE, EquiformerV2, EquiformerV3 and UMA, focusing on predicting energies and forces with high accuracy. Every architecture is trained through a single data object and a single trainer, so the choice of model is a one-word change in a configuration dictionary rather than a change of code base.


The code is available and actively maintained on `GitHub <https://github.com/changzhiai/IANN>`_. Users are encouraged to explore the repository for asking questions, or reporting issues.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   about
   installation
   quickstart
   training
   prediction
   parallelization
   lammps
   engine_models
   foundation_models
   performance
   api
   troubleshooting
   release_notes

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* :doc:`release_notes`