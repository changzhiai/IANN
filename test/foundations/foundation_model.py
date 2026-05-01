from iann.foundations import foundation_model
from iann.calculators import MLCalculator
from ase.io import read
from ase.build import fcc100

# Create calculator with the foundation model
calc = MLCalculator(
    model_path=foundation_model("painn_oc.pt"),
    compute_forces=True,  # Enable force calculations
    device='cpu'  # Use 'cuda' if you have GPU
)

# Read the last structure from the trajectory
# atoms = read('test/Pt_ads.traj', index=1)
atoms = fcc100("Pt", size=(4,4,3), a=5.5, vacuum=15.0)
# dft_energy = atoms.get_potential_energy()
# dft_forces = atoms.get_forces()

# Set the calculator
atoms.calc = calc

# Get predictions
nnp_energy = atoms.get_potential_energy()
nnp_forces = atoms.get_forces()

print(f"NNP Energy: {nnp_energy:.4f} eV")
# print(f"NNP Forces: {nnp_forces}")

# print(f"DFT Energy: {dft_energy:.4f} eV")
# print(f"DFT Forces: {dft_forces}")