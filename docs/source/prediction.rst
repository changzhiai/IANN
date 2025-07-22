Predicting Guide
===============

This guide explains how to use trained IANN models for making predictions.


Using the ML Calculator
--------------------

The ``MLCalculator`` provides a convenient ASE calculator interface:

.. code-block:: python

   from iann.calculators.calculators import MLCalculator
   from ase.io import read

   # Create calculator with model path
   calc = MLCalculator("model.pth")
   
   # Read structures
   images = read("test_structures.traj", ":")

   # Make predictions
   for atoms in images:
      atoms.calc = calc
      energy = atoms.get_potential_energy()
      forces = atoms.get_forces()
      print(f"Energy: {energy} eV")
      print(f"Forces: {forces} eV/Å")

The calculator automatically:

* Determines the model type from the saved state dict
* Use the model architecture
* Do prediction

The ``EnsembleCalculator`` provides a convenient ASE calculator interface with uncertainty for each structure:

.. code-block:: python

   from iann.calculators.calculators import EnsembleCalculator
   from ase.io import read

   # Create calculator with model path
   model_paths = ['model_1.pth', 'model_2.pth']
   calc = EnsembleCalculator(model_paths)

   # Read structures
   images = read("test_structures.traj", ":")

   # Make predictions
   for atoms in images:
      atoms.calc = calc
      energy = atoms.get_potential_energy()
      forces = atoms.get_forces()
      ensemble = atoms.calc.get_ensemble()
      print(f"Energy: {energy} eV")
      print(f"Forces: {forces} eV/Å")
      print(f"Energy variance: {ensemble['energy_var']} eV")
      print(f"Forces variance: {ensemble['forces_var']} eV/Å")

The ``AtomicEnsembleCalculator`` provides a convenient ASE calculator interface with uncertainty for each structure and each atom:

.. code-block:: python

   from iann.calculators.calculators import AtomicEnsembleCalculator
   from ase.io import read

   # Create calculator with model path
   model_paths = ['model_1.pth', 'model_2.pth']
   calc = AtomicEnsembleCalculator(model_paths)

   # Read structures
   images = read("test_structures.traj", ":")

   # Make predictions
   for atoms in images:
      atoms.calc = calc
      energy = atoms.get_potential_energy()
      forces = atoms.get_forces()
      ensemble = atoms.calc.get_ensemble()
      print(f"Energy: {energy} eV")
      print(f"Forces: {forces} eV/Å")
      print(f"Energy variance: {ensemble['energy_var']} eV")
      print(f"Forces variance: {ensemble['forces_var']} eV/Å")
      print(f"Atomic energy variance: {ensemble['atomic_energy_var']} eV")
      print(f"Atomic forces variance: {ensemble['atomic_forces_var']} eV/Å")


      
Batch Prediction
--------------

For efficient batch prediction:

.. code-block:: python

   from iann.data import AseDataset
   from torch.utils.data import DataLoader

   # Create dataset
   dataset = AseDataset("test_structures.traj")
   dataloader = DataLoader(dataset, batch_size=32)

   model = torch.load("model.pth")

   # Make predictions
   model.eval()
   for batch in dataloader:
      energy, forces = model(batch)
      # Process predictions...

Performance Tips
--------------

1. **Memory Management**

   * Use appropriate batch sizes
   * Clear GPU cache if needed

2. **Speed Optimization**

   * Enable CUDA if available
   * Use batch processing when possible
   * Consider model quantization for deployment

3. **Accuracy Considerations**

   * Check cutoff radius matches training
   * Verify atomic numbers are correct


Applications: Geometric structure optimization
------------------------------------------

The ASE optimizers ``BFGS`` like can be used to optimize the geometry of a structure:

.. code-block:: python

   from ase.optimize import BFGS
   from iann.calculators.calculators import MLCalculator

   atoms = read("atoms.traj") # load structure

   # Create calculator
   calc = MLCalculator("model.pth")
   atoms.calc = calc

   # Create optimizer
   optimizer = BFGS(atoms, trajectory="opt.traj")

   # Run optimization
   optimizer.run(fmax=0.01)


Applications: Molecular dynamics
----------------------------

The ASE thermostats like ``Langevin`` can be used to perform molecular dynamics:

.. code-block:: python  

   from ase.io import read, write, Trajectory
   from ase.md.langevin import Langevin
   from ase import units
   from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
   from ase.md import MDLogger

   atoms = read("atoms.traj") # load structure

   # Create calculator
   calc = MLCalculator("model.pth")
   atoms.calc = calc

   temperature = 300
   timestep = 0.1
   MaxwellBoltzmannDistribution(atoms, temperature_K=temperature) # initialize velocities from Maxwell-Boltzmann distribution

   # Create Langevin thermostat
   dyn = Langevin(atoms, timestep=timestep * units.fs, temperature_K=temperature, friction=0.01 / units.fs)

   # Log and save trajectory
   dyn.attach(MDLogger(dyn, atoms, 'ase_md.log', header=True, stress=False, peratom=False, mode="a"), interval=1)
   traj = Trajectory('ase_md.traj', 'a', atoms)
   dyn.attach(traj.write, interval=10)

   # Run dynamics
   dyn.run(2000)



Integration with Other Tools
-------------------------

IANN models can be used with various tools:

* ASE for other applications
* LAMMPS for molecular dynamics (see :doc:`lammps`)
* Custom scripts for specific applications

For more details on the API and advanced usage, see the :doc:`api` reference. 