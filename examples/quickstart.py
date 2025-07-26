from iann.trainer import Trainer
from iann.calculators import MLCalculator
from ase.io import read

# Initialize trainer
trainer = Trainer(
    model="painn",
    config={'device': 'cpu', 
           'output_dir': 'output',
           'output_log': 'output.log',
           'output_model': 'model.pt',
            },
    distributed=False
    )

# Train the model
trainer.train("dataset.traj")

# Create calculator with model
calc = MLCalculator("output/model.pt")

# Read structures
images = read("test_structures.traj", ":")

# Make predictions
for atoms in images:
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    print(f"Energy: {energy} eV")
    print(f"Forces: {forces} eV/Å")