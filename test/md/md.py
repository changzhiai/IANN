import os
from iann.calculators import MLCalculator
from ase.build import fcc100
from ase.io import Trajectory
from ase.md.langevin import Langevin
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import MDLogger

os.makedirs("test/md/output", exist_ok=True)

for model_type in ["painn", "nequip", "mace", "equiformerV2"]:
    print(f"\n--- Running MD with {model_type} ---")
    model_path = f"test/{model_type}/output/model.pt"
    
    if not os.path.exists(model_path):
        print(f"Skipping {model_type}: Model path {model_path} does not exist.")
        continue

    try:
        calc = MLCalculator(model_path=model_path, model_type=model_type, verbose=False)

        atoms = fcc100('Pt', size=(4,4,3), a=5.5, vacuum=15.0)
        atoms.calc = calc

        energy = atoms.get_potential_energy()
        print(f"Initial Energy: {energy:.4f} eV")

        temperature = 300
        timestep = 0.5
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
        
        dyn = Langevin(atoms, timestep=timestep * units.fs, temperature_K=temperature, friction=0.01 / units.fs)
        
        traj_file = f'test/md/output/md_{model_type}.traj'
        log_file = f'test/md/output/md_{model_type}.log'
        
        traj = Trajectory(traj_file, 'w', atoms)
        dyn.attach(MDLogger(dyn, atoms, log_file, header=True, stress=False,
                peratom=False, mode="w"), interval=1)
        dyn.attach(traj.write, interval=1)
        
        print(f"Running 50 steps of MD...")
        dyn.run(50)
        print(f"MD for {model_type} completed. Trajectory saved to {traj_file}")
        
    except Exception as e:
        print(f"Error running MD for {model_type}: {e}")
