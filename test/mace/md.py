from iann.calculators import MLCalculator
from ase.build import fcc100
from ase.io import Trajectory
from ase.md.langevin import Langevin
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# model_type = "painn"
# model_path = "test/painn/model_output/best_model.pth"

# model_type = "nequip"
# model_path = "test/nequip/model_output/best_model.pth"

model_type = "mace"
model_path = "../../model_output/best_model.pth"

# model_type = "equiformerV2"
# model_path = "test/equiformerV2/model_output/best_model.pth"

calc = MLCalculator(model_path=model_path, model_type=model_type, verbose=True)

atoms = fcc100('Pt', size=(4,4,3), a=5.5, vacuum=15.0)
atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print(energy)
print(forces)

temperature = 300
timestep = 0.1
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
# atoms.wrap()
dyn = Langevin(atoms, timestep=timestep * units.fs, temperature_K=temperature, friction=0.01 / units.fs)
traj = Trajectory('ase_md.traj', 'a', atoms)
from ase.md import MDLogger
dyn.attach(MDLogger(dyn, atoms, 'ase_md.log', header=True, stress=False,
           peratom=False, mode="a"), interval=1)
dyn.attach(traj.write, interval=1)
dyn.run(20000)
