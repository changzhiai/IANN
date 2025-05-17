import torch
from torch import nn
from iann.data.data import AtomsData, replace_properties
from typing import List, Optional
from torch import Tensor

def sinc_expansion(edge_dist: torch.Tensor, edge_size: int, cutoff: float):
    """
    calculate sinc radial basis function:
    
    sin(n *pi*d/d_cut)/d
    """
    # n tensor
    n = torch.arange(edge_size, device=edge_dist.device, dtype=edge_dist.dtype) + 1
    
    # Compute expansion
    expanded = edge_dist.unsqueeze(-1) * n * torch.pi / cutoff
    result = torch.sin(expanded) / edge_dist.unsqueeze(-1)
    
    return result

def cosine_cutoff(edge_dist: torch.Tensor, cutoff: float):
    """
    Calculate cutoff value based on distance.
    This uses the cosine Behler-Parinello cutoff function:

    f(d) = 0.5*(cos(pi*d/d_cut)+1) for d < d_cut and 0 otherwise
    """

    return torch.where(
        edge_dist < cutoff,
        0.5 * (torch.cos(torch.pi * edge_dist / cutoff) + 1),
        torch.tensor(0.0, device=edge_dist.device, dtype=edge_dist.dtype),
    )

class PainnMessage(nn.Module):
    """Message function"""
    def __init__(self, node_size: int, edge_size: int, cutoff: float):
        super().__init__()
        
        self.edge_size = edge_size
        self.node_size = node_size
        self.cutoff = cutoff
        
        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )
        
        self.filter_layer = nn.Linear(edge_size, node_size * 3)
        
    def forward(self, node_scalar, node_vector, edge_indices, edge_vectors, edge_dist):
        # remember to use v_j, s_j but not v_i, s_i        
        filter_weight = self.filter_layer(sinc_expansion(edge_dist, self.edge_size, self.cutoff))
        filter_weight = filter_weight * cosine_cutoff(edge_dist, self.cutoff).unsqueeze(-1)
        scalar_out = self.scalar_message_mlp(node_scalar)      
        filter_out = filter_weight * scalar_out[edge_indices[:, 1]]
        
        gate_state_vector, gate_edge_vector, message_scalar = torch.split(
            filter_out, 
            self.node_size,
            dim = 1,
        )
        
        # num_pairs * 3 * node_size, num_pairs * node_size
        message_vector = node_vector[edge_indices[:, 1]] * gate_state_vector.unsqueeze(1)
        edge_vector = gate_edge_vector.unsqueeze(1) * (edge_vectors / edge_dist.unsqueeze(-1)).unsqueeze(-1)
        message_vector = message_vector + edge_vector
        
        # sum message - keep contiguous() here for index_add_ operations
        residual_scalar = torch.zeros_like(node_scalar).contiguous()
        residual_vector = torch.zeros_like(node_vector).contiguous()
        residual_scalar.index_add_(0, edge_indices[:, 0], message_scalar)
        residual_vector.index_add_(0, edge_indices[:, 0], message_vector)
        
        # new node state
        new_node_scalar = node_scalar + residual_scalar
        new_node_vector = node_vector + residual_vector
        
        return new_node_scalar, new_node_vector

class PainnUpdate(nn.Module):
    """Update function"""
    def __init__(self, node_size: int):
        super().__init__()
        
        self.update_U = nn.Linear(node_size, node_size)
        self.update_V = nn.Linear(node_size, node_size)
        
        self.update_mlp = nn.Sequential(
            nn.Linear(node_size * 2, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )
        
    def forward(self, node_scalar, node_vector):
        # Linear transformations
        Uv = self.update_U(node_vector)
        Vv = self.update_V(node_vector)
        
        # Compute norm
        Vv_norm = torch.linalg.norm(Vv, dim=1)
        mlp_input = torch.cat((Vv_norm, node_scalar), dim=1)
        mlp_output = self.update_mlp(mlp_input)
        
        # Split
        a_vv, a_sv, a_ss = torch.split(
            mlp_output,                                        
            node_vector.shape[-1],                                       
            dim = 1,
        )
        
        # Compute updates
        delta_v = a_vv.unsqueeze(1) * Uv
        inner_prod = torch.sum(Uv * Vv, dim=1)
        delta_s = a_sv * inner_prod + a_ss
        
        # Return updated states
        return node_scalar + delta_s, node_vector + delta_v

class PaiNN(nn.Module):
    def __init__(
        self, 
        num_interactions, 
        hidden_state_size, 
        cutoff,
        normalization=True,
        target_mean=[0.0],
        target_stddev=[1.0],
        atomwise_normalization=True, 
        **kwargs,
    ):
        super().__init__()
        
        num_embedding = 119   # number of all elements
        self.cutoff = cutoff
        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.edge_embedding_size = 20
        
        # Setup atom embeddings
        self.atom_embedding = nn.Embedding(num_embedding, hidden_state_size)

        # Setup message-passing layers
        self.message_layers = nn.ModuleList(
            [
                PainnMessage(self.hidden_state_size, self.edge_embedding_size, self.cutoff)
                for _ in range(self.num_interactions)
            ]
        )
        self.update_layers = nn.ModuleList(
            [
                PainnUpdate(self.hidden_state_size)
                for _ in range(self.num_interactions)
            ]            
        )
        
        # Setup readout function
        self.readout_mlp = nn.Sequential(
            nn.Linear(self.hidden_state_size, self.hidden_state_size),
            nn.SiLU(),
            nn.Linear(self.hidden_state_size, 1),
        )

        # Normalisation constants
        self.normalization = torch.nn.Parameter(
            torch.tensor(normalization), requires_grad=False
        )
        self.atomwise_normalization = torch.nn.Parameter(
            torch.tensor(atomwise_normalization), requires_grad=False
        )
        self.normalize_stddev = torch.nn.Parameter(
            torch.tensor(target_stddev[0]), requires_grad=False
        )
        self.normalize_mean = torch.nn.Parameter(
            torch.tensor(target_mean[0]), requires_grad=False
        )

        self.compute_forces = False
        if 'compute_forces' in kwargs.keys():
            if kwargs['compute_forces']:
                self.compute_forces = True
                
        # Initialize parameters with proper memory layout
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reset parameters to ensure proper memory layout."""
        with torch.no_grad():
            for param in self.parameters():
                if param.requires_grad and param.dim() >= 2:
                    # Create a new tensor with the correct memory layout
                    new_data = torch.empty_like(param.data)
                    # Copy data with proper memory layout
                    new_data.copy_(param.data)
                    # Set strides to match DDP's expected layout
                    if param.dim() == 2:
                        # Create a new tensor with column-major layout
                        new_data = torch.empty(param.shape[1], param.shape[0], device=param.device, dtype=param.dtype)
                        new_data.copy_(param.data.t())
                        new_data = new_data.t()
                    param.data = new_data
                    
    def _make_contiguous(self, tensor: Optional[torch.Tensor]) -> torch.Tensor:
        """Ensure tensor has proper memory layout for DDP."""
        if tensor is None:
            raise ValueError("tensor is None in _make_contiguous")
        if tensor.dim() == 2 and tensor is not None:
            # Create a new tensor with column-major layout
            new_tensor = torch.empty(tensor.shape[1], tensor.shape[0], device=tensor.device, dtype=tensor.dtype)
            new_tensor.copy_(tensor.t())
            return new_tensor.t()
        return tensor.contiguous()
        
    def forward(self, data: AtomsData):
        """
        Args:
            data (AtomsData): A NamedTuple of model inputs

        Returns:
            AtomsData: A object with keys including 'energy' and 'forces'
        """
        num_atoms = data.num_atoms
        num_edges = data.num_edges
        positions = data.positions
        edge_indices = data.edge_indices
        edge_vectors = data.edge_vectors
        atomic_numbers = data.atomic_numbers

        if self.compute_forces:
            edge_vectors.requires_grad_()
        edge_dist = torch.linalg.norm(edge_vectors, dim=1)
        
        # Initialize node states
        node_scalar = self.atom_embedding(atomic_numbers)
        node_scalar = self._make_contiguous(node_scalar)
        
        node_vector = torch.zeros((positions.shape[0], 3, self.hidden_state_size),
                                  device=positions.device,
                                  dtype=positions.dtype)
        
        # Message passing iterations
        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            node_scalar, node_vector = message_layer(node_scalar, node_vector, edge_indices, edge_vectors, edge_dist)
            node_scalar = self._make_contiguous(node_scalar)
            node_vector = self._make_contiguous(node_vector)
            node_scalar, node_vector = update_layer(node_scalar, node_vector)

        node_scalar = self.readout_mlp(node_scalar)
        node_scalar = self._make_contiguous(node_scalar)
        node_scalar = node_scalar.squeeze()

        image_idx = torch.arange(num_atoms.shape[0],
                                 device=edge_indices.device)
        image_idx = torch.repeat_interleave(image_idx, num_atoms)
        
        # Initialize energy with proper strides
        energy = torch.zeros(num_atoms.shape[0], device=num_atoms.device, dtype=torch.float32)
        energy.index_add_(0, image_idx, node_scalar)

        # Apply (de-)normalization
        if self.normalization:
            normalizer = self.normalize_stddev
            energy = self._make_contiguous(normalizer * energy)
            mean_shift = self.normalize_mean
            if self.atomwise_normalization:
                mean_shift = self._make_contiguous(num_edges * mean_shift)
            energy = self._make_contiguous(energy + mean_shift)

        data = replace_properties(data, energy=energy)
        
        if self.compute_forces:
            # TorchScript requires explicit list types for grad arguments
            outputs_list = torch.jit.annotate(List[Tensor], [energy])
            inputs_list = torch.jit.annotate(List[Tensor], [edge_vectors])
            grad_outputs_list = torch.jit.annotate(Optional[List[Optional[Tensor]]], [torch.ones_like(energy)])
            dE_ddiff = torch.autograd.grad(
                outputs=outputs_list,
                inputs=inputs_list,
                grad_outputs=grad_outputs_list,
                retain_graph=True,
                create_graph=True,
            )[0]
            dE_ddiff = self._make_contiguous(dE_ddiff)
            
            # Initialize forces with proper strides
            i_forces = torch.zeros(positions.shape[0], 3, device=positions.device, dtype=positions.dtype)
            j_forces = torch.zeros(positions.shape[0], 3, device=positions.device, dtype=positions.dtype)
            i_forces.index_add_(0, edge_indices[:, 0], dE_ddiff)
            j_forces.index_add_(0, edge_indices[:, 1], -dE_ddiff)
            forces = self._make_contiguous(i_forces + j_forces)
            
            data = replace_properties(data, forces=forces)

        return data