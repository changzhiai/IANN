import torch
import os
import sys
from iann.data import AseDataReader
from iann.plugins.converter import convert_model_for_lammps
from iann.calculators import MLCalculator
from ase.build import fcc100

def test_model_jit(model_type, atoms):
    print(f"\n--- Testing JIT for {model_type} ---")
    
    model_path = f'test/{model_type}/output/model.pt'
    export_path = f'test/lammps_plugin/export_{model_type}.pt'
    
    if not os.path.exists(model_path):
        print(f"Skipping {model_type}: Model path {model_path} does not exist.")
        return None, None

    # Model-specific structural params if needed (matching predict scripts)
    model_kwargs = {
        'device': 'cpu',
    }
    
    if model_type == 'equiformerV2':
        model_kwargs.update({
            'grid_resolution': 12,
            'lmax_list': [4],
            'mmax_list': [2]
        })

    # 1. Get Energy from Original ASE Calculator
    try:
        calc = MLCalculator(model_path=model_path, model_type=model_type, verbose=False, **model_kwargs)
        atoms.calc = calc
        energy_ase = atoms.get_potential_energy()
    except Exception as e:
        print(f"Error in ASE calculation for {model_type}: {e}")
        return None, None

    # 2. Export to TorchScript
    try:
        # Merge model_path and model_type into kwargs for converter
        convert_params = model_kwargs.copy()
        convert_params.update({'model_path': model_path, 'model_type': model_type})
        convert_model_for_lammps(output_path=export_path, **convert_params)
    except Exception as e:
        print(f"Error exporting {model_type} to JIT: {e}")
        return energy_ase, None

    # 3. Load JIT Model and Get Energy
    try:
        scripted_model = torch.jit.load(export_path)
        cutoff = 5.5 # Default cutoff used in tests
        data_reader = AseDataReader(cutoff, compute_forces=True)
        model_inputs = data_reader(atoms)

        scripted_output = scripted_model(
            model_inputs.num_atoms,
            model_inputs.atomic_numbers,
            model_inputs.positions,
            model_inputs.cell,
            model_inputs.edge_indices,
            model_inputs.edge_vectors,
            model_inputs.num_edges,
        )
        energy_jit = scripted_output['energy'].item()
    except Exception as e:
        print(f"Error in JIT calculation for {model_type}: {e}")
        return energy_ase, None

    return energy_ase, energy_jit

if __name__ == "__main__":
    atoms = fcc100('Pt', size=(4,4,3), a=5.5, vacuum=15.0)
    
    models_to_test = ["painn", "nequip", "mace", "equiformerV2"]
    results = {}

    for model in models_to_test:
        e_ase, e_jit = test_model_jit(model, atoms)
        results[model] = (e_ase, e_jit)

    print("\n" + "="*50)
    print(f"{'Model':<15} | {'ASE Energy':<12} | {'JIT Energy':<12} | {'Diff'}")
    print("-" * 50)
    for model, (e_ase, e_jit) in results.items():
        if e_ase is not None and e_jit is not None:
            diff = abs(e_ase - e_jit)
            print(f"{model:<15} | {e_ase:>12.6f} | {e_jit:>12.6f} | {diff:>8.2e}")
        elif e_ase is not None:
            print(f"{model:<15} | {e_ase:>12.6f} | {'FAILED':<12} | N/A")
        else:
            print(f"{model:<15} | {'SKIPPED':<12} | {'SKIPPED':<12} | N/A")
    print("="*50)
