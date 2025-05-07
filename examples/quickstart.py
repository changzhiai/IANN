from iann.trainer.trainer import Trainer
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
atoms = read("test_structures.traj", ":")

# Make predictions
for atom in atoms:
    atom.calc = calc
    energy = atom.get_potential_energy()
    forces = atom.get_forces()
    print(f"Energy: {energy} eV")
    print(f"Forces: {forces} eV/Å")