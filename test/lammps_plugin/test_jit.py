import torch
from iann.data import AseDataReader
from iann.plugins.converter import convert_model_for_lammps
import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from iann.calculators import MLCalculator

def from_jit(atoms):
    convert_model_for_lammps(model_path='test/equiformerV2/output/model.pt', 
                            model_type='equiformerV2', 
                            species=['Pt'],
                            grid_resolution=12,
                            lmax_list=[4],
                            mmax_list=[2],
                            device='cpu',
                            # debug=True,
                            # atoms=atoms,
                            output_path='test/lammps_plugin/export_equiformerV2.pt')

    scripted_model = torch.jit.load('test/lammps_plugin/export_equiformerV2.pt')

    # Create test data
    cutoff = 5.5
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
    scripted_energy = scripted_output['energy'].item()
    # print(f"TorchScript energy: {scripted_energy}")
    return scripted_energy

def from_ase(atoms):
    model_type = "equiformerV2"
    model_path = "test/equiformerV2/output/model.pt"

    calc = MLCalculator(model_path=model_path, model_type=model_type, 
                grid_resolution=12,
                lmax_list=[4],
                mmax_list=[2],
                species=['Pt'],
                device='cpu')

    model_input = calc.ase_data_reader(atoms)
    # print(model_input)

    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    # print(energy)
    return energy
if __name__ == "__main__":
    from ase.build import fcc100
    atoms = fcc100('Pt', size=(4,4,3), a=5.5, vacuum=15.0)
    energy_from_jit = from_jit(atoms)
    energy_from_ase = from_ase(atoms)

    print(f"Energy from jit: {energy_from_jit}")
    print(f"Energy from ase: {energy_from_ase}")
