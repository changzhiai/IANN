from ase.io import Trajectory
import torch
from typing import List, Optional, NamedTuple, Dict, Any, Union
import asap3
import numpy as np
from scipy.spatial import distance_matrix
from e3nn import o3

class AtomsData(NamedTuple):
    num_atoms: torch.Tensor
    atomic_numbers: torch.Tensor
    positions: torch.Tensor
    cell: torch.Tensor
    edge_indices: torch.Tensor
    edge_vectors: torch.Tensor
    num_edges: torch.Tensor
    energy: Optional[torch.Tensor] = None
    forces: Optional[torch.Tensor] = None
    image_indices: Optional[torch.Tensor] = None
    atomic_energy: Optional[torch.Tensor] = None

    # nequip and mace
    atomic_types: Optional[torch.Tensor] = None
    node_attr: Optional[torch.Tensor] = None
    node_feat: Optional[torch.Tensor] = None
    edge_dist_embedding: Optional[torch.Tensor] = None
    edge_diff_embedding: Optional[torch.Tensor] = None

    # ensemble
    energy_variance: Optional[torch.Tensor] = None
    forces_variance: Optional[torch.Tensor] = None


    def to(self, device):
        new_values = {}
        for field in self._fields:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor):
                new_values[field] = value.to(device)
            else:
                new_values[field] = value
        # Return a new instance with updated values
        return self._replace(**new_values)

    def contiguous(self):
        new_values = {}
        for field in self._fields:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor):
                cont_value = value.contiguous()
                if cont_value.dim() == 2:
                    tmp = torch.empty(cont_value.shape[1], cont_value.shape[0], 
                                     device=cont_value.device, dtype=cont_value.dtype)
                    tmp.copy_(cont_value.t())
                    cont_value = tmp.t()
                new_values[field] = cont_value
            else:
                new_values[field] = value
        return self._replace(**new_values)
    
    def keys(self):
        return [field for field in self._fields if getattr(self, field) is not None]


def replace_properties(
    data: AtomsData,
    energy: Optional[torch.Tensor] = None,
    forces: Optional[torch.Tensor] = None,
    image_indices: Optional[torch.Tensor] = None,
    atomic_energy: Optional[torch.Tensor] = None,
    atomic_types: Optional[torch.Tensor] = None,
    node_attr: Optional[torch.Tensor] = None,
    node_feat: Optional[torch.Tensor] = None,
    edge_dist_embedding: Optional[torch.Tensor] = None,
    edge_diff_embedding: Optional[torch.Tensor] = None,
    energy_variance: Optional[torch.Tensor] = None,
    forces_variance: Optional[torch.Tensor] = None,
) -> AtomsData:
    return AtomsData(
        num_atoms=data.num_atoms,
        atomic_numbers=data.atomic_numbers,
        positions=data.positions,
        cell=data.cell,
        edge_indices=data.edge_indices,
        edge_vectors=data.edge_vectors,
        num_edges=data.num_edges,
        energy=energy if energy is not None else data.energy,
        forces=forces if forces is not None else data.forces,
        image_indices=image_indices if image_indices is not None else data.image_indices,
        atomic_energy=atomic_energy if atomic_energy is not None else data.atomic_energy,
        atomic_types=atomic_types if atomic_types is not None else data.atomic_types,
        node_attr=node_attr if node_attr is not None else data.node_attr,
        node_feat=node_feat if node_feat is not None else data.node_feat,
        edge_dist_embedding=edge_dist_embedding if edge_dist_embedding is not None else data.edge_dist_embedding,
        edge_diff_embedding=edge_diff_embedding if edge_diff_embedding is not None else data.edge_diff_embedding,
        energy_variance=energy_variance if energy_variance is not None else data.energy_variance,
        forces_variance=forces_variance if forces_variance is not None else data.forces_variance,
    )

class AseDataReader:
    def __init__(self, cutoff=5.0, compute_forces=False):            
        self.cutoff = cutoff
        self.compute_forces = compute_forces
        
    def __call__(self, atoms):
        num_atoms = torch.tensor([atoms.get_global_number_of_atoms()])
        atomic_numbers = torch.tensor(atoms.numbers)
        positions = torch.tensor(atoms.positions, dtype=torch.float)
        cell = torch.tensor(atoms.cell[:], dtype=torch.float)
        
        if atoms.pbc.any():
            edge_indices, edge_vectors = self.get_neighborlist(atoms)
        else:
            edge_indices, edge_vectors = self.get_neighborlist_simple(atoms)
            
        edge_indices = torch.from_numpy(edge_indices)
        edge_vectors = torch.from_numpy(edge_vectors).float()
        if self.compute_forces:
            edge_vectors.requires_grad_()
        num_edges = torch.tensor([edge_indices.shape[0]])
        
        try:
            energy = torch.tensor([atoms.get_potential_energy()], dtype=torch.float)
        except (AttributeError, RuntimeError):
            energy = None

        try: 
            forces = torch.tensor(atoms.get_forces(apply_constraint=False), dtype=torch.float)
        except (AttributeError, RuntimeError):
            forces = None
        
        # Return as AtomsData object for TorchScript compatibility
        return AtomsData(
            num_atoms=num_atoms,
            atomic_numbers=atomic_numbers,
            positions=positions,
            cell=cell,
            edge_indices=edge_indices,
            edge_vectors=edge_vectors,
            num_edges=num_edges,
            energy=energy,
            forces=forces,
            image_indices=None,
        ).contiguous()
    
    def get_neighborlist(self, atoms):        
        nl = asap3.FullNeighborList(self.cutoff, atoms)
        pair_i_idx = []
        pair_j_idx = []
        edge_vectors = []
        for i in range(len(atoms)):
            indices, diff, _ = nl.get_neighbors(i)
            pair_i_idx += [i] * len(indices)            
            pair_j_idx.append(indices)  
            edge_vectors.append(diff)

        pair_j_idx = np.concatenate(pair_j_idx)
        edge_indices = np.stack((pair_i_idx, pair_j_idx), axis=1)
        edge_vectors = np.concatenate(edge_vectors)
        
        return edge_indices, edge_vectors
    
    def get_neighborlist_simple(self, atoms):
        pos = atoms.get_positions()
        dist_mat = distance_matrix(pos, pos)
        mask = dist_mat < self.cutoff
        np.fill_diagonal(mask, False)        
        edge_indices = np.argwhere(mask)
        edge_vectors = pos[edge_indices[:, 1]] - pos[edge_indices[:, 0]]
        
        return edge_indices, edge_vectors

class AseDataset(torch.utils.data.Dataset):
    def __init__(self, ase_db, cutoff=5.0, compute_forces=False, **kwargs):
        super().__init__(**kwargs)
        
        if isinstance(ase_db, str):
            self.db = Trajectory(ase_db)
        else:
            self.db = ase_db
        
        self.cutoff = cutoff
        self.atoms_reader = AseDataReader(cutoff, compute_forces)
        
    def __len__(self):
        return len(self.db)
    
    def __getitem__(self, idx):
        atoms = self.db[idx]
        atoms_data = self.atoms_reader(atoms)
        return atoms_data

def cat_tensors(tensors: List[torch.Tensor]):
    if tensors[0].shape:
        return torch.cat(tensors)
    return torch.stack(tensors)

def collate_atomsdata(atoms_data: List[dict], pin_memory=True):
    field_names = atoms_data[0].keys()
    batched_atoms_data = AtomsData(**{
        k: torch.cat([getattr(obj, k) for obj in atoms_data if getattr(obj, k) is not None])
        for k in field_names
    })

    # Pin memory function
    pin = (lambda x: x.pin_memory()) if pin_memory else (lambda x: x)

    # create image index for each atom with proper memory layout
    image_indices = torch.repeat_interleave(
        torch.arange(len(atoms_data)), batched_atoms_data.num_atoms, dim=0
    )
    batched_atoms_data = batched_atoms_data._replace(image_indices=image_indices)
    
    # shift index of edges (because of batching) with proper memory layout
    if batched_atoms_data.edge_indices is not None:
        edge_offset = torch.zeros_like(batched_atoms_data.num_atoms)
        edge_offset[1:] = batched_atoms_data.num_atoms[:-1]
        edge_offset = torch.cumsum(edge_offset, dim=0)
        edge_offset = torch.repeat_interleave(edge_offset, batched_atoms_data.num_edges)
        edge_indices = batched_atoms_data.edge_indices + edge_offset.unsqueeze(-1)
        batched_atoms_data = batched_atoms_data._replace(edge_indices=edge_indices)
        
    return batched_atoms_data