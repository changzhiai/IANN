from iann.calculators import MLCalculator
from ase.build import fcc100

model_type = "allegro"
model_path = "test/allegro/output/model.pt"

# No architecture arguments are needed: the checkpoint written by allegro.py
# carries its own model_config, including num_scalar_features and
# num_tensor_features, and the calculator rebuilds the model from it.
calc = MLCalculator(model_path=model_path, model_type=model_type)

atoms = fcc100('Pt', size=(4,4,3), a=5.5, vacuum=15.0)

atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print(energy)
# print(forces)
