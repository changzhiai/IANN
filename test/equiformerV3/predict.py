from iann.calculators import MLCalculator
from ase.build import fcc100


model_type = "equiformerV3"
model_path = "test/equiformerV3/output/model.pt"

calc = MLCalculator(model_path=model_path, model_type=model_type,
            lmax=3,
            mmax=2,
            attn_grid_resolution_list=[12, 6],
            ffn_grid_resolution_list=[12, 12],
            device='cpu')

atoms = fcc100('Pt', size=(4, 4, 3), a=5.5, vacuum=15.0)

model_input = calc.ase_data_reader(atoms)
# print(model_input)

atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print(energy)
# print(forces)
