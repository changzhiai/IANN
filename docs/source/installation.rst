Installation
============

Prerequisites
------------

Before installing IANN, ensure you have the following prerequisites:

* Python 3.7 or higher
* PyTorch 1.9 or higher

To install PyTorch, visit the `official PyTorch website <https://pytorch.org/get-started/locally/>`_ for installation instructions specific to your system. 

Here is an example of how to install PyTorch on a Linux machine with CUDA support:

.. code-block:: bash

   # Choose Stable, Linux, Pip, Python, CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


Installing IANN
--------------

You can install IANN using pip:

.. code-block:: bash

   # Clone the repository
   git clone git@github.com:changzhiai/IANN.git
   cd IANN

   # Install with pip
   pip install .


GPU Support
----------

For GPU acceleration, make sure you have CUDA installed and PyTorch with CUDA support:

.. code-block:: bash

   # Check if PyTorch is using CUDA
   python -c "import torch; print(torch.cuda.is_available())"


Verifying Installation
--------------------

To verify your installation, you can run:

.. code-block:: python

   import iann

   # Print the version of IANN
   print(iann.__version__)

If no error occurs, IANN has been installed successfully. 


LAMMPS Interface Support
------------------------

To use IANN in LAMMPS, you need to install the LAMMPS interface. Please see LAMMPS plugins installation section :doc:`lammps` for more details.


