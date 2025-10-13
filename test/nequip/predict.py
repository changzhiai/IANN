from iann.calculators import MLCalculator
from ase.build import fcc100

model_type = "nequip"
model_path = "test/nequip/output/model.pt"

calc = MLCalculator(model_path=model_path, model_type=model_type)

from ase.build import fcc100
atoms = fcc100('Pt', size=(4,4,3), a=5.5, vacuum=15.0)

atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print(energy)
# print(forces)

