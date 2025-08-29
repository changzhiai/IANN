from iann.calculators import MLCalculator
from ase.build import fcc100
from ase.io import read
from pandas import DataFrame
from matplotlib import pyplot as plt
import numpy as np

model_type = "painn"
model_path = "test/painn/output/model.pt"
calc = MLCalculator(model_path=model_path, model_type=model_type)

# atoms = fcc100('Pt', size=(4,4,3), a=5.5, vacuum=15.0)
images = read('test/Pt_ads.traj', ':')

dft_energies = []
dft_forces = []
nnp_energies = []
nnp_forces = []

for atoms in images:
    dft_energy = atoms.get_potential_energy()
    dft_force = atoms.get_forces()
    mean_dft_force = np.mean(np.linalg.norm(dft_force, axis=1))

    atoms.calc = calc

    nnp_energy = atoms.get_potential_energy()
    nnp_force = atoms.get_forces()
    mean_nnp_force = np.mean(np.linalg.norm(nnp_force, axis=1))

    dft_energies.append(dft_energy)
    dft_forces.append(mean_dft_force)
    nnp_energies.append(nnp_energy)
    nnp_forces.append(mean_nnp_force)

df = DataFrame({
    'dft_energy': dft_energies,
    'dft_forces': dft_forces,
    'nnp_energy': nnp_energies,
    'nnp_forces': nnp_forces,
})

df.to_csv('test/painn/output/predictions.csv', index=False)

rmse = np.sqrt(np.mean((np.array(dft_energies) - np.array(nnp_energies))**2))

plt.plot(dft_energies, nnp_energies, 'o')
plt.plot([min(dft_energies), max(dft_energies)], [min(dft_energies), max(dft_energies)], 'r--')
plt.xlabel('DFT Energy')
plt.ylabel('NNP Energy')
plt.title(f'PaiNN: DFT vs NNP Energy (RMSE: {rmse:.4f})')
plt.savefig('test/painn/output/energy_comparison.png')
plt.show()

    

