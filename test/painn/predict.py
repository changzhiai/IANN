from iann.calculators.calculators import MLCalculator
from ase.build import fcc100

# model_type = "painn"
# model_path = "test/painn/model_output/best_model.pth"

# model_type = "nequip"
# model_path = "test/nequip/model_output/best_model.pth"


# model_type = "mace"
# model_path = "test/mace/model_output/best_model.pth"

model_type = "equiformerV2"
model_path = "test/equiformerV2/model_output/best_model.pth"

calc = MLCalculator(model_path=model_path, model_type=model_type)

from ase.build import fcc100
atoms = fcc100('Pt', size=(4,4,3), a=5.5, vacuum=15.0)

atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print(energy)
print(forces)

