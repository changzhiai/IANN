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


1. **Install LibTorch** from `official website <https://pytorch.org/get-started/locally/>`_: 

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

   # Make GPU version LAMMPS with LibTorch (Here is an example on NERSC. It may be different on different servers)
   module load PrgEnv-nvidia gcc cmake openmpi cudatoolkit # Load required modules on NERSC
   GPU_ARCH=`nvidia-smi --query-gpu=name --format=csv,noheader`
   cmake ../cmake \
   -D CMAKE_PREFIX_PATH=$INSTALL_PATH/libtorch \
   -D CMAKE_CXX_FLAGS="-I$INSTALL_PATH/libtorch/include/torch/csrc/api/include -I$INSTALL_PATH/libtorch/include"  \
   -D Torch_DIR=$INSTALL_PATH/libtorch/share/cmake/Torch \
   -D CMAKE_BUILD_TYPE=Release \
   -D PKG_GPU=yes  \
   -D GPU_API=cuda \
   -D GPU_ARCH=$GPU_ARCH \
   -D PKG_USER-MISC=ON \
   -D BUILD_MPI=ON \ 
   -D BUILD_OMP=ON \
   -D CMAKE_EXE_LINKER_FLAGS="-L$INSTALL_PATH/libtorch/lib -Wl,-rpath,$INSTALL_PATH/libtorch/lib -ltorch -ltorch_cpu -lc10"
   make -j 8

If you want to make CPU version LAMMPS with LibTorch rather than GPU version, you can use the following command:

.. code-block:: bash

   # Make CPU version LAMMPS with LibTorch (Here is an example on S3DF. It may be different on different servers)
   module use /sdf/scratch/users/c/changzhi/softwares/easybuild/modules/all # Load easybuild modules on S3DF
   module load GCC/11.3.0 CMake/3.23.1-GCCcore-11.3.0 OpenMPI # Load required modules on S3DF
   cmake ../cmake \
   -D CMAKE_PREFIX_PATH=$INSTALL_PATH/libtorch \
   -D CMAKE_CXX_FLAGS="-I$INSTALL_PATH/libtorch/include/torch/csrc/api/include -I$INSTALL_PATH/libtorch/include" \
   -D Torch_DIR=$INSTALL_PATH/libtorch/share/cmake/Torch \
   -D PKG_USER-MISC=ON \
   -D BUILD_MPI=ON \
   -D BUILD_OMP=ON \
   -D CMAKE_EXE_LINKER_FLAGS="-L$INSTALL_PATH/libtorch/lib -Wl,-rpath,$INSTALL_PATH/libtorch/lib -ltorch -ltorch_cpu -lc10" 
   make -j 8


Usage of LAMMPS with IANN models
-------------------------------

1. **Convert a trained model to the torchscript format**:

First, you need to have a trained model with torch format, which can be obtained by running the training script. Then convert the model to the torchscript format as follows ``convert.py``:

.. code-block:: python

   from iann.plugins.converter import convert_model_for_lammps

   # Convert the model to the torchscript format
   convert_model_for_lammps(model_path='model.pt', 
                           model_type='painn', # if not specified, the model type will be inferred from the model file
                           output_path='model_lmp.pt')

2. **Use the exported model in LAMMPS**:

Here's a basic LAMMPS input script to use the exported model, the input file is structure file ``initial.data`` and model file ``model_lmp.pt``.


To convert ase trajectory to lammps data file, you can use the following script:

.. code-block:: python

   from ase.io import read
   from ase.io.lammpsdata import write_lammps_data

   # Read the initial structure
   atoms = read('start.traj')

   # Write the initial structure to the lammps data file
   write_lammps_data("initial.data", atoms, masses=True)


To run the LAMMPS simulation, you can use the following script ``in.lmp``:

.. code-block:: lammps

   # LAMMPS input script example
   
   # Define the units and the atom style
   units metal
   atom_style atomic

   # Define the boundary conditions
   boundary p p p

   # Read the initial structure
   read_data initial.data

   # Define the IANN pair style
   pair_style iann painn model_lmp.pt 5.5
   pair_coeff * *
   
   # Define the mass of the atoms
   mass 1 1.0079999997406976 # H
   mass 2 195.08399994981576 # Pt

   # Define the neighbor list
   neighbor 0.5 bin
   neigh_modify every 1 delay 0 check yes

   # Thermodynamic settings
   thermo 10

   # Initial minimization to relax the system before dynamics
   minimize 1.0e-4 1.0e-6 100 1000

   # Define the timestep and the thermostat
   timestep 0.001
   fix 1 all nvt temp 300.0 300.0 0.1

   # Define the dump frequency and the dump file
   dump 1 all custom 10 dump.xyz id type x y z

   # Run the simulation
   run 5000

Output would be log file ``lammps.log`` and structures file ``dump.xyz``. 

.. note::

   Recommended to run the LAMMPS simulation on GPU, which is much faster than CPU. But you need to use the GPU version of LAMMPS, which can be installed by the previous installation section.


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

   # Read the structures from the xyz file
   images = read('dump.xyz', ':')

   # Define the type to symbol mapping
   type_to_symbol = {1: 1, 2: 78,} # H and Pt

   for atoms in images:
      # Get the atom types
      atom_types = atoms.numbers

      # Convert the atom types to atomic numbers
      atomic_numbers = [type_to_symbol[t] for t in atom_types]

      # Set the atomic numbers
      atoms.set_atomic_numbers(atomic_numbers)

   # Write the structures to the trajectory file
   write('trajectory.traj', images)

   # Visualize the structures
   view(images)


The log file ``lammps.log`` is shown something like below:

.. code-block:: text

   Step  Temp          E_pair         E_mol          TotEng         Press 
      0   0             -504.91425      0             -504.91425     -27751.64 
      1   0             -511.008        0             -511.008       -22307.96 
      2   0             -515.71405      0             -515.71405     -19594.152
      3   0             -519.0155       0             -519.0155      -15338.803
      4   0             -521.05121      0             -521.05121     -11473.622
      5   0             -522.61688      0             -522.61688     -4825.5281
      6   0             -523.04858      0             -523.04858     -5309.5876
      7   0             -523.29437      0             -523.29437     -4772.4662
      8   0             -523.68579      0             -523.68579     -1643.6579
      9   0             -523.78748      0             -523.78748     -1175.3389
      10   0             -523.88159      0             -523.88159     -1221.8421
      11   0             -524.00598      0             -524.00598     -886.14601
      12   0             -524.06201      0             -524.06201      170.3822 



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

1. **Convert trained models to a ensemble model**:

First, you need to have several trained models with torch format, which can be obtained by running several training scripts. Then convert the models to the torchscript format as follows:

.. code-block:: python

   from iann.plugins.converter import convert_models_for_lammps

   # Give a list of models
   model_paths = ["model_1.pt", "model_2.pt"]

   # Convert the models to a torchscript model
   output_path = convert_models_for_lammps(
       model_paths=model_paths,
       model_type="painn", # if not specified, the model type will be inferred from the model file
       output_path="model_ensemble_lmp.pt"
   )

2. **Use the exported ensemble model in LAMMPS**:

Here's a basic LAMMPS input script ``in.lmp`` to use the exported ensemble model:

.. code-block:: lammps

   # LAMMPS input script example
   
   # Define the units and the atom style
   units metal
   atom_style atomic

   # Define the boundary conditions
   boundary p p p

   # Read the initial structure
   read_data initial.data

   # Define the IANN pair style
   pair_style iann painn model_ensemble_lmp.pt 5.5
   pair_coeff * *

   # Define the mass of the atoms
   mass 1 1.0079999997406976 # H
   mass 2 195.08399994981576 # Pt

   # Define the neighbor list
   neighbor 0.5 bin
   neigh_modify every 1 delay 0 check yes

   # Compute the variance mode of the energy and force of the ensemble model
   compute variance all iann/variance
   
   # Define the thermodynamic style
   thermo_style custom step pe ke etotal temp press c_variance[1] c_variance[2] c_variance[3] c_variance[4]

   # Define the thermodynamic modify
   thermo_modify colname c_variance[1] energy_var
   thermo_modify colname c_variance[2] force_var
   thermo_modify colname c_variance[3] max_energy_var
   thermo_modify colname c_variance[4] max_force_var
   thermo_modify flush yes

   # Thermodynamic settings
   thermo 100

   # Initial minimization to relax the system before dynamics
   minimize 1.0e-4 1.0e-6 100 1000

   # Define the timestep and the thermostat
   timestep 0.001
   fix 1 all nvt temp 300.0 300.0 0.1

   # Define the dump frequency and the dump file
   dump 1 all custom 10 dump.xyz id type x y z

   # Run the simulation
   run 5000

Output would be ``lammps.log`` and ``dump.xyz``. You can use the same script ``view.py`` to visualize the structures.

The log file ``lammps.log`` is shown something like below:

.. code-block:: text

   Step    PotEng         KinEng         TotEng          Temp          Press        energy_var     force_var   max_energy_var   max_force_var
   0    -511.36917      3.1410215     -508.22815      300           -25080.902      3.3947108      0.97444135     0.014607213    0.38785329
   100  -517.70813      9.2895191     -508.41861      887.24503      9948.8746      0.34052402     6.9993577      0.01117126     0.13307451
   200  -515.76294      6.7511453     -509.01179      644.80411     -13103.606      1.3004521      0.54851878     0.0092412131   0.28447273
   300  -523.90332      13.670955     -510.23237      1305.7174     -3064.5944      0.013090299    0.0361195      0.025595488    0.13039519
   400  -523.13562      11.832817     -511.3028       1130.1563     -984.54894      0.058141664    0.21462031     0.023984909    0.098044716
   500  -524.34436      12.574164     -511.7702       1200.9626      6828.6078      0.026319327    0.78461003     0.029852344    0.12279015
   600  -523.93469      11.237698     -512.69699      1073.3162      3937.144       0.10693619     0.57325053     0.018862754    0.23991443
   700  -524.31494      10.969936     -513.34501      1047.7422      7551.752       0.037434116    0.42069691     0.021637097    0.099797592
   800  -525.68494      11.14164      -514.5433       1064.1417      9043.1032      0.0017779488   0.2508451      0.018614301    0.1285743
   900  -524.61243      9.1504548     -515.46197      873.96295      12192.979      0.10897815     0.8868334      0.056080569    0.29281014
   1000  -526.6048       10.167        -516.4378       971.05348      5777.5752      0.10106332     0.41264692     0.0096192099   0.084062219
   1100  -525.99164      8.865721      -517.12592      846.76793      12156.906      0.18633902     0.97626913     0.022417877    0.32700533
   1200  -524.8175       7.3447002     -517.4728       701.49474      12890.364      0.086737752    0.67208624     0.0091473255   0.19820641


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


Usage of LAMMPS with IANN models on multiple GPUs
------------------------------------------------

Multiple GPUs prediction (inference) is supported by using the ``pair_style iann/multi_gpu`` command. It will automatically detect the number of GPUs per node and use them to run the model. Here is an ``in.lmp`` example of how to use it:

.. code-block:: lammps

   # LAMMPS input script example of running on multiple GPUs

   # Define the units and the atom style
   units metal
   atom_style atomic

   # Define the boundary conditions
   boundary p p p

   # Read the initial structure
   read_data initial.data

   # Define the IANN pair style
   pair_style iann/multi_gpu painn model_lmp.pt 5.5
   pair_coeff * *

   # Define the mass of the atoms
   mass 1 1.0079999997406976 # H
   mass 2 195.08399994981576 # Pt

   # Define the neighbor list
   neighbor 0.5 bin
   neigh_modify every 1 delay 0 check yes

   # Thermodynamic settings
   thermo 10

   # Initial minimization to relax the system before dynamics
   minimize 1.0e-4 1.0e-6 100 1000

   # Define the timestep and the thermostat
   timestep 0.001
   fix 1 all nvt temp 300.0 300.0 0.1

   # Define the dump frequency and the dump file
   dump 1 all custom 10 dump.xyz id type x y z

   # Run the simulation
   run 5000

The key components are the same as the single GPU inference version, and we just need to replace the ``pair_style iann`` command with the ``pair_style iann/multi_gpu`` command. The log file ``lammps.log`` is shown exactly the same as the single GPU inference version. 


Usage of LAMMPS with IANN models with ensemble on multiple GPUs
--------------------------------------------------------------
Multiple GPUs prediction (inference) is supported by using the ``pair_style iann/multi_gpu`` and ``compute variance all iann/variance`` command. It will automatically detect the number of GPUs per node and use them to run the ensemble model. Here is an ``in.lmp`` example of how to use it:

.. code-block:: lammps

   # LAMMPS input script example of running on multiple GPUs with ensemble

   # Define the units and the atom style
   units metal
   atom_style atomic

   # Define the boundary conditions
   boundary p p p

   # Read the initial structure
   read_data initial.data

   # Define the IANN pair style
   pair_style iann/multi_gpu painn model_ensemble_lmp.pt 5.5
   pair_coeff * *

   # Define the mass of the atoms
   mass 1 1.0079999997406976 # H
   mass 2 195.08399994981576 # Pt

   # Define the neighbor list
   neighbor 0.5 bin
   neigh_modify every 1 delay 0 check yes

   # Compute the variance mode of the energy and force of the ensemble model
   compute variance all iann/variance

   # Thermodynamic settings
   thermo 10
   thermo_style custom step pe ke etotal press c_variance[1] c_variance[2] c_variance[3] c_variance[4]
   thermo_modify colname c_variance[1] energy_variance
   thermo_modify colname c_variance[2] force_variance
   thermo_modify colname c_variance[3] max_energy_variance
   thermo_modify colname c_variance[4] max_force_variance

   # Initial minimization to relax the system before dynamics
   minimize 1.0e-4 1.0e-6 100 1000

   # Define the timestep and the thermostat
   timestep 0.001
   fix 1 all nvt temp 300.0 300.0 0.1

   # Define the dump frequency and the dump file
   dump 1 all custom 10 dump.xyz id type x y z

   # Run the simulation
   run 5000

The key components are the same as the single GPU inference version with ensemble, and we just need to replace the ``pair_style iann`` command with the ``pair_style iann/multi_gpu`` command. And add the ``compute variance all iann/variance`` command to compute the variance of the energy and force of the ensemble model. The log file ``lammps.log`` is shown exactly the same as the single GPU inference version with ensemble.

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