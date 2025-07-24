from iann.trainer import Trainer
from iann.calculators.calculators import MLCalculator
from ase.io import read

# Train a model
trainer = Trainer(
    model="painn",
    config={"device": "cpu", 
            'output_dir': 'output'},
    distributed=False
    )
trainer.train("dataset.traj")

# Create calculator with model path
calc = MLCalculator("output/best_model.pth")

# Read structures
images = read("test_structures.traj", ":")

# Make predictions
for atoms in images:
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    print(f"Energy: {energy} eV")
    print(f"Forces: {forces} eV/Å")