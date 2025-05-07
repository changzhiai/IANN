from iann.models.painn import PainnModel
from iann.trainer.trainer import Trainer
import torch
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

# Load model
state_dict = torch.load("best_model.pth")
model = PainnModel(
    num_interactions=state_dict["num_layer"],
    hidden_state_size=state_dict["node_size"],
    cutoff=state_dict["cutoff"],
    compute_forces=True
)
model.load_state_dict(state_dict["model"])

# Create calculator
calc = MLCalculator(model)

# Predict
atoms = read("test_structures.traj", ":")
for atom in atoms:
    atom.calc = calc
    energy = atom.get_potential_energy()
    forces = atom.get_forces()
    print(f"Energy: {energy} eV")
    print(f"Forces: {forces} eV/Å")