#!/usr/bin/env python3
"""
Script to export trained IANN models to TorchScript format for LAMMPS integration.

Converts trained PyTorch models to TorchScript format
Creates a wrapper that adapts model inputs/outputs for LAMMPS
Supports all four model types with proper error handling
"""

import argparse
import torch
from pathlib import Path
from iann.data.data import AseDataReader, AtomsData
from typing import Dict

class LAMMPSModelWrapper(torch.nn.Module):
    def __init__(self, model, compute_forces=True):
        super().__init__()
        self.model = model
        self.compute_forces = compute_forces

    def forward(self,
                num_atoms: torch.Tensor,
                atomic_numbers: torch.Tensor,
                positions: torch.Tensor,
                cell: torch.Tensor,
                edge_indices: torch.Tensor,
                edge_vectors: torch.Tensor,
                num_edges: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward seven input tensors matching the C++ plugin call."""
        # Reconstruct the NamedTuple internally
        model_inputs = AtomsData(
            num_atoms=num_atoms,
            atomic_numbers=atomic_numbers,
            positions=positions,
            cell=cell,
            edge_indices=edge_indices,
            edge_vectors=edge_vectors,
            num_edges=num_edges,
            energy=None,
            forces=None,
            image_indices=None,
        )
        if self.compute_forces:
            model_inputs.edge_vectors.requires_grad_()
        data = self.model(model_inputs)
        if not hasattr(self.model, 'compute_forces') or not self.model.compute_forces:
            raise RuntimeError("Model did not return forces. Make sure compute_forces=True")

        results = {}
        if data.energy is not None:
            results['energy'] = data.energy
        if data.forces is not None:
            results['forces'] = data.forces
        if data.atomic_energy is not None:
            results['atomic_energy'] = data.atomic_energy
            
        return results
    
class EnsembleLAMMPSModelWrapper(torch.nn.Module):
    def __init__(self, models, compute_forces=True):
        super().__init__()
        self.models = models
        self.compute_forces = compute_forces

    def forward(self,
                num_atoms: torch.Tensor,
                atomic_numbers: torch.Tensor,
                positions: torch.Tensor,
                cell: torch.Tensor,
                edge_indices: torch.Tensor,
                edge_vectors: torch.Tensor,
                num_edges: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass that computes ensemble averages and variances."""
        # Reconstruct the NamedTuple internally
        model_inputs = AtomsData(
            num_atoms=num_atoms,
            atomic_numbers=atomic_numbers,
            positions=positions,
            cell=cell,
            edge_indices=edge_indices,
            edge_vectors=edge_vectors,
            num_edges=num_edges,
            energy=None,
            forces=None,
            image_indices=None,
        )
        if self.compute_forces:
            model_inputs.edge_vectors.requires_grad_()

        # Collect predictions from all models
        all_energies = []
        all_forces = []
        all_atomic_energies = []
        for model in self.models:
            data = model(model_inputs)
            if not hasattr(model, 'compute_forces') or not model.compute_forces:
                raise RuntimeError("Model did not return forces. Make sure compute_forces=True")
            energy = data.energy
            forces = data.forces
            atomic_energies = data.atomic_energy
            assert energy is not None
            assert forces is not None
            assert atomic_energies is not None
            all_energies.append(energy)
            all_forces.append(forces)
            all_atomic_energies.append(atomic_energies)
            
        # Calculate ensemble averages and variances
        avg_energy = torch.mean(torch.stack(all_energies), dim=0)
        avg_forces = torch.mean(torch.stack(all_forces), dim=0)
        
        # Calculate variances
        energy_var = torch.var(torch.stack(all_energies), dim=0)
        forces_var = torch.var(torch.stack(all_forces), dim=0)
        atomic_energy_var = torch.var(torch.stack(all_atomic_energies), dim=0)
        
        results = {}
        if avg_energy is not None:
            results['energy'] = avg_energy
        if avg_forces is not None:
            results['forces'] = avg_forces
        if energy_var is not None:
            results['energy_variance'] = energy_var
        if forces_var is not None:
            results['forces_variance'] = forces_var
        if atomic_energy_var is not None:
            results['atomic_energy_variance'] = atomic_energy_var
        return results

def convert_model_for_lammps(model_path, model_type, output_path=None, compute_forces=True):
    """Wrap a trained model in a TorchScript-compatible wrapper for LAMMPS.
    
    Args:
        model_path (str): Path to the trained model checkpoint
        model_type (str): Type of model (painn, nequip, mace, equiformer2)
        output_path (str, optional): Path to save the exported model
    
    Returns:
        str: Path to the exported TorchScript model
    """
    print(f"Loading model from {model_path}")
    
    # Load the model checkpoint
    device = torch.device('cpu')
    state_dict = torch.load(model_path, map_location=device)
    
    # Create appropriate model wrapper based on type
    if model_type.lower() == "painn":
        from iann.models.painn import PaiNN
        node_size = state_dict.get("node_size", 128)
        num_interactions = state_dict.get("num_layer", 3)
        cutoff = state_dict.get("cutoff", 5.5)
        raw_model = PaiNN(
            hidden_state_size=node_size,
            num_interactions=num_interactions,
            cutoff=cutoff,
            compute_forces=True,
            normalization=False,
            atomwise_normalization=False,
        )
        raw_model.load_state_dict(state_dict["model"])
    elif model_type.lower() == "nequip":
        try:
            from iann.models.nequip import NequIP
            num_interactions = state_dict.get("num_layer", 3)
            node_size = state_dict.get("node_size", 128)
            cutoff = state_dict.get("cutoff", 5.5)
            raw_model = NequIP(
                num_interactions=num_interactions,
                num_features=node_size,
                cutoff=cutoff,
                compute_forces=True
            )
            raw_model.load_state_dict(state_dict["model"])
        except ImportError:
            raise ImportError("Nequip is not available")
    elif model_type.lower() == "mace":
        try:
            from iann.models.mace import MACE
            num_interactions = state_dict.get("num_layer", 3)
            node_size = state_dict.get("node_size", 128)
            cutoff = state_dict.get("cutoff", 5.5)
            raw_model = MACE(
                num_interactions=num_interactions,
                num_features=node_size,
                cutoff=cutoff,
                correlation = 3,
                species = None,
                compute_forces=True
            )
            raw_model.load_state_dict(state_dict["model"])
        except ImportError:
            raise ImportError("MACE is not available")
    elif model_type.lower() == "equiformerv2":
        try:
            from iann.models.equiformerV2 import EquiformerV2
            num_interactions = state_dict.get("num_layer", 3)
            node_size = state_dict.get("node_size", 128)
            cutoff = state_dict.get("cutoff", 5.5)
            raw_model = EquiformerV2(
                num_interactions=num_interactions,
                num_features=node_size,
                cutoff=cutoff,
                compute_forces=True
            )
            raw_model.load_state_dict(state_dict["model"])
        except ImportError:
            raise ImportError("EquiformerV2 is not available")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    wrapped_model = LAMMPSModelWrapper(raw_model, compute_forces=compute_forces)
    wrapped_model.eval()
    # Example test: verify the wrapper with dummy ASE atoms
    example = False
    if example:
        from ase.build import fcc100
        test_atoms = fcc100('Pt', size=(4,4,3), a=5.5, vacuum=15.0)
        model_inputs = AseDataReader(cutoff, compute_forces=compute_forces)(test_atoms)
        example_out = wrapped_model(
            model_inputs.num_atoms,
            model_inputs.atomic_numbers,
            model_inputs.positions,
            model_inputs.cell,
            model_inputs.edge_indices,
            model_inputs.edge_vectors,
            model_inputs.num_edges,
        )
        print(f"Example test passed: Energy={example_out.energy}, Forces shape={example_out.forces.shape}")

    scripted_model = torch.jit.script(wrapped_model)
    
    if output_path is None:
        output_path = f"{Path(model_path).stem}_{model_type}_lammps.pt"
    torch.jit.save(scripted_model, output_path)
    print(f"Model exported to {output_path}")
    return output_path

def convert_models_for_lammps(model_paths, model_type, output_path=None, compute_forces=True):
    """Convert multiple models to a single TorchScript model for LAMMPS with ensemble statistics.
    
    Args:
        model_paths (list): List of paths to trained model checkpoints
        model_type (str): Type of model (painn, nequip, mace, equiformer2)
        output_path (str, optional): Path to save the exported model
    
    Returns:
        str: Path to the exported TorchScript model
    """
    print(f"Loading {len(model_paths)} models for ensemble")
    
    # Load all models
    device = torch.device('cpu')
    models = torch.nn.ModuleList()
    
    for model_path in model_paths:
        print(f"Loading model from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        
        # Create appropriate model based on type
        if model_type.lower() == "painn":
            from iann.models.painn import PaiNN
            node_size = state_dict.get("node_size", 128)
            num_interactions = state_dict.get("num_layer", 3)
            cutoff = state_dict.get("cutoff", 5.5)
            raw_model = PaiNN(
                hidden_state_size=node_size,
                num_interactions=num_interactions,
                cutoff=cutoff,
                compute_forces=True,
                normalization=False,
                atomwise_normalization=False,
            )
        elif model_type.lower() == "nequip":
            from iann.models.nequip import NequIP
            num_interactions = state_dict.get("num_layer", 3)
            node_size = state_dict.get("node_size", 128)
            cutoff = state_dict.get("cutoff", 5.5)
            raw_model = NequIP(
                num_interactions=num_interactions,
                num_features=node_size,
                cutoff=cutoff,
                compute_forces=True
            )
        elif model_type.lower() == "mace":
            from iann.models.mace import MACE
            num_interactions = state_dict.get("num_layer", 3)
            node_size = state_dict.get("node_size", 128)
            cutoff = state_dict.get("cutoff", 5.5)
            raw_model = MACE(
                num_interactions=num_interactions,
                num_features=node_size,
                cutoff=cutoff,
                correlation=3,
                species=None,
                compute_forces=True
            )
        elif model_type.lower() == "equiformerv2":
            from iann.models.equiformerV2 import EquiformerV2
            num_interactions = state_dict.get("num_layer", 3)
            node_size = state_dict.get("node_size", 128)
            cutoff = state_dict.get("cutoff", 5.5)
            raw_model = EquiformerV2(
                num_interactions=num_interactions,
                num_features=node_size,
                cutoff=cutoff,
                compute_forces=True
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        raw_model.load_state_dict(state_dict["model"])
        raw_model.eval()
        models.append(raw_model)

    # Create ensemble wrapper
    wrapped_model = EnsembleLAMMPSModelWrapper(models, compute_forces=compute_forces)
    wrapped_model.eval()

    # Test the wrapper with dummy ASE atoms
    from ase.build import fcc100
    test_atoms = fcc100('Pt', size=(4,4,3), a=5.5, vacuum=15.0)
    model_inputs = AseDataReader(models[0].cutoff, compute_forces=compute_forces)(test_atoms)

    # Script the model
    scripted_model = torch.jit.script(wrapped_model)
    
    if output_path is None:
        output_path = f"ensemble_{model_type}_lammps.pt"
    torch.jit.save(scripted_model, output_path)
    print(f"Ensemble model exported to {output_path}")
    return output_path 


def main():
    parser = argparse.ArgumentParser(description="Export IANN models to TorchScript for LAMMPS")
    parser.add_argument("--model_path", "-m", required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--model_type", "-t", choices=["painn", "nequip", "mace", "equiformer2"], required=True,
                        help="Type of model to export")
    parser.add_argument("--output", "-o", help="Output path for exported model")
    
    args = parser.parse_args()
    
    convert_model_for_lammps(args.model_path, args.model_type, args.output)

if __name__ == "__main__":
    main()

