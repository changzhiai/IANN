Installation
============

Prerequisites
------------

Before installing IANN, ensure you have the following prerequisites:

* ASE 3.24 or higher
* PyTorch 1.9 or higher
* Python 3.7 or higher
* ASAP3 3.13 or higher
* e3nn 0.4.4 or higher

Installing IANN
--------------

You can install IANN using pip:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/changzhiai/IANN.git
   cd IANN

   # Install with pip
   pip install .


GPU Support
----------

For GPU acceleration, make sure you have CUDA installed and PyTorch with CUDA support:

.. code-block:: bash

   # Check if PyTorch is using CUDA
   python -c "import torch; print(torch.cuda.is_available())"

If you need to install PyTorch with CUDA support, visit the `official PyTorch website <https://pytorch.org/get-started/locally/>`_ for installation instructions specific to your system.

Verifying Installation
--------------------

To verify your installation, you can run:

.. code-block:: python

   import iann
   print(iann.__version__)

If no error occurs, IANN has been installed successfully. 