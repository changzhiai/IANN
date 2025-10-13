from iann.calculators import MLCalculator
from ase.build import fcc100


model_type = "equiformerV2"
model_path = "test/equiformerV2/output/model.pt"

calc = MLCalculator(model_path=model_path, model_type=model_type, 
            grid_resolution=12,
            lmax_list=[4],
            mmax_list=[2],
            species=['Pt'],
            device='cpu')

from ase.build import fcc100
atoms = fcc100('Pt', size=(4,4,3), a=5.5, vacuum=15.0)

model_input = calc.ase_data_reader(atoms)
# print(model_input)

atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print(energy)
# print(forces)

