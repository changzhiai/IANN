LAMMPS Interface
==============

This guide explains how to use IANN models as interatomic potentials in LAMMPS molecular dynamics simulations.


Installation
------------

**Prerequisites**

* LibTorch (PyTorch C++ API) v1.8.0 or later
* LAMMPS (with C++11 support or later)
* GCC >= 7.1
* CMake >= 3.18
* OpenMPI


1. **Install LibTorch** from official website`<https://pytorch.org/get-started/locally/>`_: 

Here is an example that selects Stable, Linux, LibTorch, C++/Java, and CUDA 11.8. Then downloading as follows: 

.. code-block:: bash

   # Choose Stable, Linux, LibTorch, C++/Java, and CUDA 11.8
   INSTALL_PATH=~/softwares # Change to your own directory
   cd $INSTALL_PATH
   wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu118.zip
   unzip libtorch-cxx11-abi-shared-with-deps-2.7.1+cu118.zip
   cd libtorch


2. **Install LAMMPS intergrated with libtorch**

Install GPU version LAMMPS with LibTorch:

.. code-block:: bash

   # Clone LAMMPS to a local directory, and copy the IANN plugin files to the LAMMPS source code, and build the LAMMPS
   cd $INSTALL_PATH
   git clone https://github.com/lammps/lammps.git
   cd lammps/src
   cp $INSTALL_PATH/IANN/iann/plugins/*.h $INSTALL_PATH/IANN/iann/plugins/*.cpp .
   cd .. && mkdir build && cd build

   # Make GPU version LAMMPS with LibTorch (Here an example on NERSC. It may be different on different servers)
   module load PrgEnv-nvidia gcc cmake openmpi cudatoolkit # Load required modules on NERSC
   GPU_ARCH=`nvidia-smi --query-gpu=name --format=csv,noheader`
   cmake ../cmake -DCMAKE_PREFIX_PATH=$INSTALL_PATH/libtorch \
   -DCMAKE_CXX_FLAGS="-I$INSTALL_PATH/libtorch/include/torch/csrc/api/include -I$INSTALL_PATH/libtorch/include"   \
   -DTorch_DIR=$INSTALL_PATH/libtorch/share/cmake/Torch \
   -DCMAKE_BUILD_TYPE=Release -DPKG_GPU=yes  -DGPU_API=cuda -DGPU_ARCH=$GPU_ARCH \
   -DPKG_USER-MISC=ON -DBUILD_MPI=ON   -DBUILD_OMP=ON   \
   -DCMAKE_EXE_LINKER_FLAGS="-L$INSTALL_PATH/libtorch/lib -Wl,-rpath,$INSTALL_PATH/libtorch/lib -ltorch \
   -ltorch_cpu -lc10" ; make -j 8

If you want to make CPU version LAMMPS with LibTorch rather than GPU version, you can use the following command:

.. code-block:: bash

   cmake ../cmake -DCMAKE_PREFIX_PATH=$INSTALL_PATH/libtorch \
   -DCMAKE_CXX_FLAGS="-I$INSTALL_PATH/libtorch/include/torch/csrc/api/include -I$INSTALL_PATH/libtorch/include"   \
   -DTorch_DIR=$INSTALL_PATH/libtorch/share/cmake/Torch \
   -DCMAKE_BUILD_TYPE=Release -DPKG_GPU=no  -DGPU_API=cuda -DGPU_ARCH=$GPU_ARCH \
   -DPKG_USER-MISC=ON -DBUILD_MPI=ON   -DBUILD_OMP=ON   \
   -DCMAKE_EXE_LINKER_FLAGS="-L$INSTALL_PATH/libtorch/lib -Wl,-rpath,$INSTALL_PATH/libtorch/lib -ltorch \
   -ltorch_cpu -lc10" ; make -j 8


Usage of LAMMPS with IANN models
-------------------------------

1. **To use an IANN model with LAMMPS**, first export a trained model with torch format to the torchscript format:

First, you need to have a trained model with torch format, which can be obtained by running the training script. Then convert the model to the torchscript format as follows ``convert.py``:

.. code-block:: python

    from iann.plugins.converter import convert_model_for_lammps

    convert_model_for_lammps(model_path='model.pth', 
                            model_type='painn', # if not specified, the model type will be inferred from the model file
                            output_path='model_lmp.pth')

2. **Use the exported model in LAMMPS**:

Here's a basic LAMMPS input script to use the exported model, the input file is structure file ``initial.data`` and model file ``model_lmp.pt``.


To convert ase trajectory to lammps data file, you can use the following script:

.. code-block:: python

   from ase.io import read
   from ase.io.lammpsdata import write_lammps_data

   atoms = read('start.traj')
   write_lammps_data("initial.data", atoms, masses=True)


To run the LAMMPS simulation, you can use the following script ``in.lmp``:

.. code-block:: lammps

   # LAMMPS input script example
   units metal
   atom_style atomic
   boundary p p p

   read_data initial.data

   # Define the IANN pair style
   pair_style iann painn model_lmp.pt 5.5
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

Output would be log file ``lammps.log`` and structures file ``dump.xyz``. 

Key Components:

1. **Units and Style**

   * Use ``units metal`` for eV and Å
   * ``atom_style atomic`` for basic atomic systems

2. **Model Setup**

   .. code-block:: lammps

      pair_style iann model_type model_name.pt cutoff_radius

   * specify model type, e.g., ``painn``, ``nequip``, ``mace``, ``equiformerV2``
   * specify the model file name, e.g., ``model_lmp.pt``
   * specify the cutoff radius, e.g., ``5.5`` Å

3. **MD Settings**

   * Choose appropriate ensemble (NVE, NVT, NPT)
   * Set suitable timestep (typically 0.001 fs)
   * Set thermo (print the thermodynamic information every 100 steps)
   * Set temperature (e.g., 300.0 K)
   * Configure output frequency (e.g., 10)

4. **Run on GPU/CPU**

   * It supports running on GPU/CPU.
   * It will automatically detect the GPU/CPU. If there is no GPU, it will run on CPU.

To convert xyz file to ase trajectory and visualize the structures, you can use the following script ``view.py``:

.. code-block:: python

   from ase.io import read, write
   from ase.visualize import view

   images = read('dump.xyz', ':')
   type_to_symbol = {1: 1, 2: 78,} # H and Pt

   for atoms in images:
      atom_types = atoms.numbers
      atomic_numbers = [type_to_symbol[t] for t in atom_types]
      atoms.set_atomic_numbers(atomic_numbers)

   write('trajectory.traj', images)
   view(images)

An submission example script of running LAMMPS with IANN model on SLURM:

.. code-block:: bash

   #!/bin/bash
   #SBATCH -N 1                   # Number of nodes
   #SBATCH -C gpu                 # Use GPU nodes
   #SBATCH -q debug               # Use regular/debug queue
   #SBATCH -t 00:30:00            # Time limit
   #SBATCH -A m2997               # Your account
   #SBATCH --gpus-per-node=1      # GPUs per node
   #SBATCH --ntasks-per-node=1    # Number of tasks per node
   #SBATCH --cpus-per-task=1      # Number of CPUs per task

   module purge; module load PrgEnv-nvidia; module load openmpi; module load cudatoolkit/11.7
   lmp -in in.md

Usage of LAMMPS with ensemble IANN models
----------------------------------------

1. **To use an ensemble IANN model with LAMMPS**, first export trained models with torch format to a ensemble model with the torchscript format:

First, you need to have several trained models with torch format, which can be obtained by running several training scripts. Then convert the models to the torchscript format as follows:

.. code-block:: python

   from iann.plugins.converter import convert_models_for_lammps

   model_paths = ["model_1.pth", "model_2.pth"]
   output_path = convert_models_for_lammps(
       model_paths=model_paths,
       model_type="painn", # if not specified, the model type will be inferred from the model file
       output_path="./model_ensemble_lmp.pth"
   )

2. **Use the exported ensemble model in LAMMPS**:

Here's a basic LAMMPS input script ``in.lmp`` to use the exported ensemble model:

.. code-block:: lammps

   # LAMMPS input script example
   units metal
   atom_style atomic
   boundary p p p

   read_data initial.data

   # Define the IANN pair style
   pair_style iann painn model_ensemble_lmp.pth 5.5
   pair_coeff * *

   mass 1 1.0079999997406976 # H
   mass 2 195.08399994981576 # Pt

   neighbor 0.5 bin
   neigh_modify every 1 delay 0 check yes

   compute variance all iann/variance
   thermo_style custom step pe ke etotal temp press c_variance[1] c_variance[2] c_variance[3] c_variance[4]
   thermo_modify colname c_variance[1] energy_var
   thermo_modify colname c_variance[2] force_var
   thermo_modify colname c_variance[3] max_energy_var
   thermo_modify colname c_variance[4] max_force_var
   thermo_modify flush yes

   # Thermodynamic settings
   thermo 100

   # Initial minimization to relax the system before dynamics
   minimize 1.0e-4 1.0e-6 100 1000

   # Run your simulation
   timestep 0.001
   fix 1 all nvt temp 300.0 300.0 0.1
   dump 1 all custom 10 dump.xyz id type x y z

    run 5000

Output would be ``lammps.log`` and ``dump.xyz``.

Key Components:

1. **Ensemble Model Setup**

   .. code-block:: lammps

      pair_style iann model_type ensemble_model_name.pt cutoff_radius
   
   * specify model type, e.g., ``painn``, ``nequip``, ``mace``, ``equiformerV2``
   * specify the ensemble model file name, e.g., ``model_ensemble_lmp.pt``
   * specify the cutoff radius, e.g., ``5.5`` Å

2. **Compute Variance**

   Compute the variance of the energy and force of the ensemble model:

   .. code-block:: lammps

      compute variance all iann/variance 
      thermo_style custom step pe ke etotal temp press c_variance[1] c_variance[2] c_variance[3] c_variance[4]
      thermo_modify colname c_variance[1] energy_var
      thermo_modify colname c_variance[2] force_var
      thermo_modify colname c_variance[3] max_energy_var
      thermo_modify colname c_variance[4] max_force_var
      thermo_modify flush yes
   
   * Use ``compute variance all iann/variance`` to compute the variance of the energy and force of the ensemble model.
   * Use ``thermo_style custom step pe ke etotal temp press c_variance[1] c_variance[2] c_variance[3] c_variance[4]`` to print the variance of the energy and force of the ensemble model.
   * Use ``thermo_modify colname c_variance[1] energy_var`` to change the name of c_variance[1] to energy_var.
   * Use ``thermo_modify colname c_variance[2] force_var`` to change the name of c_variance[2] to force_var.
   * Use ``thermo_modify colname c_variance[3] max_energy_var`` to change the name of c_variance[3] to max_energy_var.
   * Use ``thermo_modify colname c_variance[4] max_force_var`` to change the name of c_variance[4] to max_force_var.
   * Use ``thermo_modify flush yes`` to flush the thermodynamic output.


Troubleshooting
-------------

1. **Model Loading**

   * Verify model file exists
   * Check file permissions
   * Ensure correct format

2. **Performance Issues**

   * Adjust neighbor list settings
   * Monitor memory usage

3. **Accuracy Concerns**

   * Verify cutoff radius
   * Check unit

For more details on LAMMPS usage and advanced features, refer to the `LAMMPS documentation <https://docs.lammps.org/>`_. 