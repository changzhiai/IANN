import numpy as np
import os
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from iann.tools import calc_energy_statistics

def create_synthetic_dataset(n_images=100):
    """
    Creates a synthetic dataset where energy is a linear combination of atom counts:
    E = n_H * (-1.0) + n_O * (-3.0) + n_Si * (-5.0) + noise
    And forces have different RMS per type.
    """
    images = []
    # Ground truth shifts
    gt_shifts = {'H': -1.0, 'O': -3.0, 'Si': -5.0}
    # Ground truth scales (relative RMS)
    gt_rms = {'H': 0.5, 'O': 1.5, 'Si': 1.0}
    
    for i in range(n_images):
        n_h = np.random.randint(1, 5)
        n_o = np.random.randint(1, 5)
        n_si = np.random.randint(1, 5)
        
        symbols = ['H'] * n_h + ['O'] * n_o + ['Si'] * n_si
        n_atoms = len(symbols)
        
        atoms = Atoms(symbols, positions=np.random.randn(n_atoms, 3))
        
        # Energy
        energy = n_h * gt_shifts['H'] + n_o * gt_shifts['O'] + n_si * gt_shifts['Si']
        energy += np.random.normal(0, 0.01) # Add tiny noise
        
        # Forces with different RMS per type
        forces = np.zeros((n_atoms, 3))
        idx = 0
        for s in ['H', 'O', 'Si']:
            count = {'H': n_h, 'O': n_o, 'Si': n_si}[s]
            forces[idx:idx+count] = np.random.normal(0, gt_rms[s], size=(count, 3))
            idx += count
            
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        images.append(atoms)
        
    return images, gt_shifts, gt_rms

def test_calc_energy_statistics():
    print("Generating synthetic dataset...")
    images, gt_shifts, gt_rms = create_synthetic_dataset(200)
    
    # 1. Test with explicit species
    print("\nTest 1: Explicit species")
    species = ['H', 'O', 'Si']
    stats = calc_energy_statistics(images, species=species, log_file="test_explicit.log")
    
    for s in species:
        recovered = stats['per_type_energy_shifts'][s]
        expected = gt_shifts[s]
        diff = abs(recovered - expected)
        print(f"  {s} shift: recovered={recovered:.4f}, expected={expected:.4f}, diff={diff:.4e}")
        assert diff < 0.1, f"Shift recovery failed for {s}"

    # 2. Test with auto-discovery
    print("\nTest 2: Auto-discovery")
    stats_auto = calc_energy_statistics(images, species=None, log_file="test_auto.log")
    assert sorted(stats_auto['species']) == sorted(['H', 'O', 'Si'])
    print(f"  Auto-discovered species: {stats_auto['species']}")

    # 3. Test sampling
    print("\nTest 3: Sampling")
    stats_sample = calc_energy_statistics(images, max_samples=50, log_file="test_sample.log")
    # Shifts should still be reasonably close
    for s in species:
        recovered = stats_sample['per_type_energy_shifts'][s]
        expected = gt_shifts[s]
        print(f"  {s} shift (sampled): recovered={recovered:.4f}, expected={expected:.4f}")
        assert abs(recovered - expected) < 0.5

    # 4. Check scales
    print("\nTest 4: Scales (Force RMS)")
    all_rms = np.array([gt_rms[s] for s in ['H', 'O', 'Si']])
    mean_rms = np.mean(all_rms)
    expected_scales = {s: gt_rms[s] / mean_rms for s in ['H', 'O', 'Si']}
    
    for s in species:
        recovered = stats['per_type_energy_scales'][s]
        expected = expected_scales[s]
        diff = abs(recovered - expected)
        print(f"  {s} scale: recovered={recovered:.4f}, expected={expected:.4f}, diff={diff:.4e}")
        assert diff < 0.2, f"Scale recovery failed for {s}"

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_calc_energy_statistics()
