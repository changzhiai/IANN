from iann.data import AtomsData, replace_properties
import torch
from torch import nn
from typing import List, Optional, Tuple
from torch import Tensor
from e3nn import o3
from e3nn.o3 import Linear, TensorProduct, FullyConnectedTensorProduct
from e3nn.nn import FullyConnectedNet, Gate, NormActivation
import math
import abc
def sinc_expansion(edge_dist: torch.Tensor, edge_size: int, cutoff: float):
    """
    Calculate sinc radial basis function:
    
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

def get_triplet_indices(edge_indices: torch.Tensor, num_atoms: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate triplet indices for 3-body interactions.
    Returns (i, j, k) indices where i is the central atom.
    """
    # Create all possible triplets
    triplets = []
    
    # For each atom i, find all pairs of neighbors (j, k)
    for i in range(num_atoms):
        # Find all edges where i is the central atom
        i_edges = torch.where(edge_indices[:, 0] == i)[0]
        
        if len(i_edges) >= 2:  # Need at least 2 neighbors for 3-body
            for j_idx in range(len(i_edges)):
                for k_idx in range(j_idx + 1, len(i_edges)):
                    j = edge_indices[i_edges[j_idx], 1]
                    k = edge_indices[i_edges[k_idx], 1]
                    triplets.append([i, j, k])
    
    if len(triplets) == 0:
        # Return empty tensors if no triplets
        return torch.empty((0,), dtype=torch.long, device=edge_indices.device), \
               torch.empty((0,), dtype=torch.long, device=edge_indices.device), \
               torch.empty((0,), dtype=torch.long, device=edge_indices.device)
    
    triplets = torch.tensor(triplets, device=edge_indices.device)
    return triplets[:, 0], triplets[:, 1], triplets[:, 2]

class HOTMEMRadialBasis(nn.Module):
    """HOTMEM-style radial basis functions"""
    def __init__(self, num_basis: int, cutoff: float, power: int = 6):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        self.power = power
        
        # Learnable basis parameters
        self.basis_params = nn.Parameter(torch.randn(num_basis))
        
    def forward(self, edge_dist: torch.Tensor) -> torch.Tensor:
        # Bessel basis functions
        n = torch.arange(1, self.num_basis + 1, device=edge_dist.device, dtype=edge_dist.dtype)
        basis = torch.sin(n * torch.pi * edge_dist.unsqueeze(-1) / self.cutoff) / edge_dist.unsqueeze(-1)
        
        # Apply learnable weights
        basis = basis * self.basis_params.unsqueeze(0)
        
        # Apply cutoff
        cutoff_fn = torch.where(
            edge_dist < self.cutoff,
            torch.pow(1 - edge_dist / self.cutoff, self.power),
            torch.zeros_like(edge_dist)
        )
        
        return basis * cutoff_fn.unsqueeze(-1)

class Message(nn.Module):
    """Enhanced message function with O3NN features for 2-body interactions"""
    def __init__(self, embeddings: dict, num_channels: int):
        super().__init__()
        
        self.embeddings = embeddings
        self.node_dim = embeddings['node'].num_channels
        self.edge_dim = embeddings['edge'].num_channels
        self.sh_dim = embeddings['angular'].num_channels
        self.global_dim = embeddings['global'].num_channels
        self.num_channels = num_channels

        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.num_channels),
            nn.SiLU(),
            nn.Linear(self.num_channels, self.num_channels * 1),
        )
        
        self.edge_message_mlp = nn.Sequential(
            nn.Linear(self.edge_dim, self.num_channels),
            nn.SiLU(),
            nn.Linear(self.num_channels, self.num_channels * 1),
        )

        self.angular_message_mlp = nn.Sequential(
            nn.Linear(self.sh_dim, self.num_channels),
            nn.SiLU(),
            nn.Linear(self.num_channels, self.num_channels * 1),
        )

        
    def forward(self, scalar_features, vector_features, edge_features, angular_features, global_features, edge_indices, edge_vectors, edge_dist):

        node_in = self.scalar_message_mlp(scalar_features[edge_indices[:, 1]])  

        edge_mlp = self.edge_message_mlp(edge_features)

        angular_mlp = self.angular_message_mlp(angular_features)

        global_in = global_features[edge_indices[:, 1]]

        
        # Weight spherical harmonics by edge magnitude (equivariant)
        # angular_mlp = angular_mlp * edge_dist.unsqueeze(-1) # [num_edges, sh_dim] x [num_edges, 1]
        
        # edge_mix = angular_in * edge_dist.unsqueeze(-1) / 5.5
        edge_in = angular_mlp * edge_mlp  # [num_edges, sh_dim, 3]

        global_pass = global_in * edge_in * node_in

        edge_pass = global_pass * edge_in * node_in

        node_pass = global_pass * edge_pass * node_in

        residual_scalar = torch.zeros_like(scalar_features)
        residual_scalar.index_add_(0, edge_indices[:, 0], node_pass)

        # edge_mix = edge_mix * node_in
        

        # gate_state_vector, gate_edge_vector, message_scalar = torch.split(
        #     edge_mix, 
        #     self.num_channels,
        #     dim = 1,
        # )

        # message_vector = vector_features[edge_indices[:, 1]] * gate_state_vector.unsqueeze(1)
        # edge_vector = gate_edge_vector.unsqueeze(1) * (edge_vectors / edge_dist.unsqueeze(-1)).unsqueeze(-1)
        # message_vector = message_vector + edge_vector
        
        # residual_scalar = torch.zeros_like(scalar_features)
        # residual_vector = torch.zeros_like(vector_features)
        # residual_scalar.index_add_(0, edge_indices[:, 0], message_scalar)
        # residual_vector.index_add_(0, edge_indices[:, 0], message_vector)
        
        # # new node state
        # new_node_scalar = scalar_features + residual_scalar
        # new_node_vector = vector_features + residual_vector
        
        # return new_node_scalar, new_node_vector
        return residual_scalar

class Update(nn.Module):
    """Enhanced update function with O3NN and spherical harmonics"""
    def __init__(self, num_channels: int):
        super().__init__()
        
        # Update MLPs
        # self.update_U = nn.Linear(num_channels, num_channels)
        
        # self.update_V = nn.Linear(num_channels, num_channels)
        
        # self.update_mlp = nn.Sequential(
        #     nn.Linear(num_channels * 2, num_channels),
        #     nn.SiLU(),
        #     nn.Linear(num_channels, num_channels * 3),
        # )
        
        # # Combine features for final update
        # self.combine_update = nn.Sequential(
        #     nn.Linear(num_channels * 2, num_channels),
        #     nn.SiLU(),
        #     nn.Linear(num_channels, num_channels),
        # )
        
    def forward(self, node_features, node_pass):
        # Linear transformations
        # Uv = self.update_U(force_vector)
        # Vv = self.update_V(force_vector)
        
        # # Compute norm
        # Vv_norm = torch.linalg.norm(Vv, dim=1)
        # mlp_input = torch.cat((Vv_norm, node_features), dim=1)
        # mlp_output = self.update_mlp(mlp_input)
        
        # a_vv, a_sv, a_ss = torch.split(
        #     mlp_output,                                        
        #     force_vector.shape[-1],                                       
        #     dim = 1,
        # )
        
        # # Compute updates
        # delta_v = a_vv.unsqueeze(1) * Uv
        # inner_prod = torch.sum(Uv * Vv, dim=1)
        # delta_s = a_sv * inner_prod + a_ss
        
        # Return updated states
        # return node_features + delta_s, force_vector + delta_v

        return node_features + node_pass
    
class RadialBasis(torch.nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self):
        pass

class BesselBasis(RadialBasis):
    def __init__(self, cutoff: float, num_basis: int=8, trainable: bool=True):
        r"""Radial Bessel Basis, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


        Parameters
        ----------
        cutoff : float
            Cutoff radius

        num_basis : int
            Number of Bessel Basis functions

        trainable : bool
            Train the :math:`n \pi` part or not.
        """
        super(BesselBasis, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis

        self.cutoff = float(cutoff)
        self.prefactor = 2.0 / self.cutoff
        # output edge dist irreps
        self.irreps_out = o3.Irreps([(num_basis, o3.Irrep(0, 1))])

        bessel_weights = (
            torch.linspace(start=1.0, end=num_basis, steps=num_basis) * math.pi
        )
        if self.trainable:
            self.bessel_weights = nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Bessel Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1) / self.cutoff)

        return self.prefactor * (numerator / x.unsqueeze(-1))

def _poly_cutoff(x: torch.Tensor, factor: float, p: float = 6.0) -> torch.Tensor:
    x = x * factor

    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
    out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))

    return out * (x < 1.0)

class CutoffFunction(torch.nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self):
        pass

class PolynomialCutoff(CutoffFunction):
    def __init__(self, cutoff: float, power: float = 6):
        r"""Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


        Parameters
        ----------
        cutoff : float
            Cutoff radius

        power : int
            Power used in envelope function
        """
        super().__init__()
        assert power >= 2.0
        self.p = float(power)
        self._factor = 1.0 / float(cutoff)

    def forward(self, x):
        """
        Evaluate cutoff function.

        x: torch.Tensor, input distance
        """
        return _poly_cutoff(x, self._factor, p=self.p)

class NodeEmbedding(nn.Module):
    """Node embedding module"""
    def __init__(self, num_channels: int, species: List[int]):
        super().__init__()
        self.species = species
        self.num_channels = num_channels

        if self.species is None:
            self.num_embedding = 119
        else:
            self.num_embedding = len(self.species)

        self.atom_embedding = nn.Embedding(self.num_embedding, self.num_channels)

        self.irreps_out = o3.Irreps(f"{self.num_embedding}x0e")

    def forward(self, data: AtomsData) -> AtomsData:
        node_attr = self.atom_embedding(data.atomic_numbers)
        data = replace_properties(data, node_attr=node_attr)

        return data

class EdgeEmbedding(nn.Module):
    """Edge embedding module"""
    def __init__(self, basis: nn.Module, cutoff_fn: nn.Module):  
        super().__init__()
        self.basis = basis
        self.cutoff_fn = cutoff_fn
        self.num_channels = basis.num_basis

        # output edge dist irreps
        self.irreps_out = self.basis.irreps_out

    def forward(self, data: AtomsData) -> AtomsData:
        edge_dist = torch.linalg.norm(data.edge_vectors, dim=1)
        edge_dist_embedding = (self.basis(edge_dist) * self.cutoff_fn(edge_dist)[:, None])
        data = replace_properties(data, edge_dist_embedding=edge_dist_embedding)

        return data

class AngularEmbedding(nn.Module):
    """Angular embedding module"""
    def __init__(self, edge_sh_irreps: o3.Irreps, edge_sh_normalize: bool=True, edge_sh_normalization: str='component'):
        super().__init__()
        self.sh = o3.SphericalHarmonics(edge_sh_irreps, edge_sh_normalize, edge_sh_normalization)
        self.num_channels = edge_sh_irreps.dim
        # output edge diff irreps
        self.irreps_out = edge_sh_irreps

    def forward(self, data: AtomsData) -> AtomsData:
        edge_diff_embedding = self.sh(data.edge_vectors)
        data = replace_properties(data, edge_diff_embedding=edge_diff_embedding)
        
        return data

class GlobalEmbedding(nn.Module):
    """Global embedding module"""
    def __init__(self, batch_size: int, num_channels: int, ):
        super().__init__()
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.global_embedding = nn.Sequential(
            nn.Linear(8, num_channels),
            nn.SiLU(),
            nn.Linear(num_channels, num_channels),
        )
        
    def forward(self, data: AtomsData) -> AtomsData:
        global_attr = data.global_attr
        global_attr = global_attr.reshape(-1, 8)
        global_attr = torch.repeat_interleave(global_attr, data.num_atoms, dim=0)
        global_embedding = self.global_embedding(global_attr)
        data = replace_properties(data, global_embedding=global_embedding)
        
        return data


class HOTMEM(nn.Module):
    """
    HOTMEM: High-Order Tensor Multi-body Equivariant Message passing neural network.
    
    An enhanced neural network model that combines the power of PaiNN with O3NN features
    and multi-body interactions beyond traditional 2-body approaches.
    """
    def __init__(
        self, 
        num_layers=3, 
        num_channels=128, 
        norm_data=True,
        data_mean=[0.0],
        data_stddev=[1.0],
        norm_per_atom=True, 
        **kwargs,
    ):
        """
        Initialize the HOTMEM model.
        
        Parameters
        ----------
        num_layers : int
            Number of message passing layers
        num_channels : int
            Number of feature channels
        norm_data : bool
            Whether to normalize the data
        data_mean : list
            Mean values for data normalization
        data_stddev : list
            Standard deviation values for data normalization
        norm_per_atom : bool
            Whether to normalize per atom
        **kwargs : dict
            Additional keyword arguments including:
            - num_embedding: Number of atomic embeddings (default: 119)
            - cutoff: Interaction cutoff distance (default: 5.5)
            - edge_embedding_size: Size of edge embeddings (default: 20)
            - lmax: Maximum spherical harmonic degree (default: 2)
            - use_multi_body: Enable multi-body interactions (default: True)
            - compute_forces: Compute forces during inference (default: False)
        """
        super().__init__()
        
        self.cutoff = kwargs.get('cutoff', 5.5)
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.lmax = kwargs.get('lmax', 2)
        self.use_multi_body = kwargs.get('use_multi_body', False)
        self.species = kwargs.get('species', None)
        self.num_basis: int = kwargs.get('num_basis', 8)
        self.power: int = kwargs.get('power', 6)
        self.batch_size = kwargs.get('batch_size', 12)
        

        self.embeddings = nn.ModuleDict()
        # node embedding
        self.embeddings['node'] = NodeEmbedding(self.num_channels, self.species)
        # edge embedding
        self.basis = BesselBasis(cutoff=self.cutoff, num_basis=self.num_basis)
        self.cutoff_fn = PolynomialCutoff(cutoff=self.cutoff, power=self.power)
        self.embeddings['edge'] = EdgeEmbedding(self.basis, self.cutoff_fn)
        # angular embedding
        self.edge_sh_irreps = o3.Irreps.spherical_harmonics(self.lmax, p=-1)
        self.embeddings['angular'] = AngularEmbedding(self.edge_sh_irreps)
        # global embedding
        self.embeddings['global'] = GlobalEmbedding(self.batch_size, self.num_channels)

        # Setup message-passing layers
        self.message_layers = nn.ModuleList([
            Message(self.embeddings, self.num_channels)
            for _ in range(self.num_layers)
        ])
        
        # Setup multi-body message layers
        # if self.use_multi_body:
        #     self.multi_body_layers = nn.ModuleList([
        #         MultiBodyMessage(self.embeddings, self.num_channels)
        #         for _ in range(self.num_layers)
        #     ])
        
        self.update_layers = nn.ModuleList([
            Update(self.num_channels)
            for _ in range(self.num_layers)
        ]            
        )
        
        # Setup readout function
        self.readout_mlp = nn.Sequential(
            nn.Linear(self.num_channels, self.num_channels),
            nn.SiLU(),
            nn.Linear(self.num_channels, 1),
        )

        # Normalisation constants
        self.norm_data = torch.nn.Parameter(
            torch.tensor(norm_data), requires_grad=False
        )
        self.norm_per_atom = torch.nn.Parameter(
            torch.tensor(norm_per_atom), requires_grad=False
        )
        self.normalize_stddev = torch.nn.Parameter(
            torch.tensor(data_stddev[0]), requires_grad=False
        )
        self.data_mean = torch.nn.Parameter(
            torch.tensor(data_mean[0]), requires_grad=False
        )

        self.compute_forces = False
        if 'compute_forces' in kwargs.keys():
            if kwargs['compute_forces']:
                self.compute_forces = True
        

        
    def forward(self, data: AtomsData):
        """
        Forward pass with enhanced features.
        
        Parameters
        ----------
        data : AtomsData
            Input data containing atomic information and edge connectivity
            
        Returns
        -------
        AtomsData
            Output data with predicted energies and optionally forces
        """
        num_atoms = data.num_atoms
        num_edges = data.num_edges
        positions = data.positions
        edge_indices = data.edge_indices
        atomic_numbers = data.atomic_numbers
        edge_vectors = data.edge_vectors
        edge_dist = torch.linalg.norm(edge_vectors, dim=1)
        
        # Process embeddings following MACE pattern
        for embedding in self.embeddings.values():
            data = embedding(data)
        
        # Extract embeddings from data
        node_features = data.node_attr  # One-hot atomic features
        edge_features = data.edge_dist_embedding  # Radial basis features
        angular_features = data.edge_diff_embedding  # Spherical harmonics features
        global_features = data.global_embedding  # Global features
        force_vector = torch.zeros((positions.shape[0], 3, self.num_channels), device=positions.device, dtype=torch.float32)
        
        # Message passing iterations
        for layer_idx in range(self.num_layers):
            # 2-body interactions
            node_pass = self.message_layers[layer_idx](
                node_features, force_vector, edge_features, angular_features, global_features, edge_indices, edge_vectors, edge_dist
            )
            
            # Update step
            node_features = self.update_layers[layer_idx](node_features, node_pass)

        # Readout
        node_features = self.readout_mlp(node_features)
        node_features = node_features.squeeze()

        # Aggregate atomic energies
        image_idx = torch.arange(num_atoms.shape[0], device=edge_indices.device)
        image_idx = torch.repeat_interleave(image_idx, num_atoms)
        
        energy = torch.zeros(num_atoms.shape[0], device=num_atoms.device, dtype=torch.float32)
        energy.index_add_(0, image_idx, node_features)

        atomic_energy = node_features
        data = replace_properties(data, atomic_energy=atomic_energy)

        # Apply normalization
        if self.norm_data:
            normalizer = self.normalize_stddev
            energy = normalizer * energy
            mean_shift = self.data_mean
            if self.norm_per_atom:
                mean_shift = num_edges * mean_shift
            energy = energy + mean_shift

        data = replace_properties(data, energy=energy)
        
        # Force computation
        if self.compute_forces:
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
            
            # Initialize forces
            i_forces = torch.zeros(positions.shape[0], 3, device=positions.device, dtype=torch.float32)
            j_forces = torch.zeros(positions.shape[0], 3, device=positions.device, dtype=torch.float32)
            i_forces.index_add_(0, edge_indices[:, 0], dE_ddiff)
            j_forces.index_add_(0, edge_indices[:, 1], -dE_ddiff)
            forces = i_forces + j_forces
            
            data = replace_properties(data, forces=forces)

        return data 