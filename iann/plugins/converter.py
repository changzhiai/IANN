#!/usr/bin/env python3
"""
Script to export trained IANN models to TorchScript format for LAMMPS integration.

Converts trained PyTorch models to TorchScript format
Creates a wrapper that adapts model inputs/outputs for LAMMPS
Supports all four model types with proper error handling
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from iann.data.data import AseDataReader, AtomsData
import asap3


def safe_replace(obj, **kwargs):
    if hasattr(obj, "_replace"):
        return obj._replace(**kwargs)
    else:
        # Manual fallback: rebuild new instance
        d = obj.__dict__.copy()
        d.update(kwargs)
        return type(obj)(**d)

class LAMMPSModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.compute_forces = True

    def forward(self,
                num_atoms: torch.Tensor,
                atomic_numbers: torch.Tensor,
                positions: torch.Tensor,
                cell: torch.Tensor,
                edge_indices: torch.Tensor,
                edge_vectors: torch.Tensor,
                num_edges: torch.Tensor) -> AtomsData:
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
        return data

def convert_model_for_lammps(model_path, model_type, output_path=None):
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

    wrapped_model = LAMMPSModelWrapper(raw_model)
    wrapped_model.eval()
    # Example test: verify the wrapper with dummy ASE atoms
    from ase.build import fcc100
    test_atoms = fcc100('Pt', size=(4,4,3), a=5.5, vacuum=15.0)
    model_inputs = AseDataReader(cutoff, compute_forces=True)(test_atoms)

    # Unpack for the new forward signature
    # example_out = wrapped_model(
    #     model_inputs.num_atoms,
    #     model_inputs.atomic_numbers,
    #     model_inputs.positions,
    #     model_inputs.cell,
    #     model_inputs.edge_indices,
    #     model_inputs.edge_vectors,
    #     model_inputs.num_edges,
    # )
    # print(f"Example test passed: Energy={example_out.energy}, Forces shape={example_out.forces.shape}")

    scripted_model = torch.jit.script(wrapped_model)

    # scripted_model = torch.jit.trace(wrapped_model, (model_inputs.num_atoms,
    #     model_inputs.atomic_numbers,
    #     model_inputs.positions,
    #     model_inputs.cell,
    #     model_inputs.edge_indices,
    #     model_inputs.edge_vectors,
    #     model_inputs.num_edges,))
    
    if output_path is None:
        output_path = f"{Path(model_path).stem}_{model_type}_lammps.pt"
    torch.jit.save(scripted_model, output_path)
    print(f"Model exported to {output_path}")
    return output_path


def get_neighborlist(atoms, cutoff):        
    nl = asap3.FullNeighborList(cutoff, atoms)
    pair_i_idx = []
    pair_j_idx = []
    edge_vectors_list = []
    for i in range(len(atoms)):
        indices, diff, _ = nl.get_neighbors(i)
        pair_i_idx += [i] * len(indices)
        pair_j_idx.append(indices)
        edge_vectors_list.append(diff)

    pair_j_idx = np.concatenate(pair_j_idx)
    edge_indices = np.stack((pair_i_idx, pair_j_idx), axis=1)
    edge_vectors = np.concatenate(edge_vectors_list)
    
    return edge_indices, edge_vectors


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