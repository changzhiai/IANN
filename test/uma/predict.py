from iann.calculators import MLCalculator
from ase.build import fcc100

model_type = "uma"
model_path = "test/uma/output/model.pt"

calc = MLCalculator(
    model_path=model_path,
    model_type=model_type,
    lmax=2,
    mmax=2,
    hidden_channels=32,
    edge_channels=32,
    num_distance_basis=128,
    norm_type="rms_norm_sh",
    device="cpu",
)

atoms = fcc100("Pt", size=(4, 4, 3), a=5.5, vacuum=15.0)

atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print(energy)
# print(forces)
