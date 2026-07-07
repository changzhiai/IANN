from iann.data import AtomsData, replace_properties
import torch
from torch import nn
from typing import List, Optional, Tuple
from torch import Tensor
from e3nn import o3
import math
import abc

class Interaction(nn.Module):
    """Enhanced message function with O3NN features for 2-body interactions"""
    def __init__(self, embeddings: dict, num_channels: int):
        super().__init__()
        
        self.embeddings = embeddings
        self.node_dim = embeddings['node'].num_channels
        self.edge_dim = embeddings['edge'].num_channels
        self.sh_dim = embeddings['angular'].num_channels
        # self.global_dim = embeddings['global'].num_channels
        self.num_channels = num_channels

        self.msg_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.num_channels),
            nn.SiLU(),
            nn.Linear(self.num_channels, self.num_channels * 1),
        )

        self.edge_linear = nn.Linear(self.edge_dim, self.num_channels * 1)
        self.angular_linear = nn.Linear(self.sh_dim, self.num_channels * 1)

    def forward(self, node_features, edge_features, angular_features, edge_indices):
        node_in = self.msg_mlp(node_features[edge_indices[:, 1]]) 
        edge_mlp = self.edge_linear(edge_features)
        angular_mlp = self.angular_linear(angular_features)
        edge_in = angular_mlp * edge_mlp
        
        node_pass = node_in * edge_in

        residual_node = torch.zeros_like(node_features, device=edge_indices.device)
        residual_node.index_add_(0, edge_indices[:, 0], node_pass)

        node_features = node_features + residual_node

        return node_features.contiguous()
    
class RadialBasis(torch.nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self):
        pass

class BesselBasis(RadialBasis):
    def __init__(self, cutoff: float, num_basis: int=8, trainable: bool=True):
        r"""
        Radial Bessel Basis, as proposed in DimeNet: https://arxiv.org/abs/2003.03123

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
        self.global_embedding = nn.Linear(8, num_channels * 1)
        
    def forward(self, data: AtomsData) -> AtomsData:
        global_attr = data.global_attr
        global_attr = global_attr.reshape(-1, 8)
        global_attr = torch.repeat_interleave(global_attr, data.num_atoms, dim=0)
        global_embedding = self.global_embedding(global_attr)
        data = replace_properties(data, global_embedding=global_embedding)
        
        return data


class Demo(nn.Module):
    """
    Demo: Demo model (High-Order Tensor Equivariant Message Passing Neural Network).
    
    A high-performance neural network model that combines the power of high-order tensor features
    and equivariant message passing for demo energy surface prediction.
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
            - compute_forces: Compute forces during inference (default: False)
        """
        super().__init__()
        
        self.cutoff = kwargs.get('cutoff', 5.5)
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.lmax = kwargs.get('lmax', 7)
        self.use_multi_body = kwargs.get('use_multi_body', False)
        self.species = kwargs.get('species', None)
        self.num_basis: int = kwargs.get('num_basis', num_channels)
        self.power: int = kwargs.get('power', 6)
        self.batch_size = kwargs.get('batch_size', 12)
        self.forces_scale = kwargs.get('forces_scale', 1.0)
        

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
        # self.embeddings['global'] = GlobalEmbedding(self.batch_size, self.num_channels)

        # Setup message-passing layers
        self.interaction_layers = nn.ModuleList([
            Interaction(self.embeddings, self.num_channels) for _ in range(self.num_layers)
        ])     
        
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
        edge_vectors = data.edge_vectors
        
        # Process embeddings following MACE pattern
        for embedding in self.embeddings.values():
            data = embedding(data)
        
        # Extract embeddings from data
        node_features = data.node_attr  # One-hot atomic features
        edge_features = data.edge_dist_embedding  # Radial basis features
        angular_features = data.edge_diff_embedding  # Spherical harmonics features
        # global_features = data.global_embedding  # Global features
        
        force_vector = torch.zeros((positions.shape[0], 3, self.num_channels), device=positions.device, dtype=torch.float32)
        
        # Message passing iterations
        for layer_idx in range(self.num_layers):
            node_features = self.interaction_layers[layer_idx](node_features, edge_features, angular_features, edge_indices)
        
        # Readout
        node_features = self.readout_mlp(node_features)
        node_features = node_features.squeeze()
        
        # Ensure node_features is contiguous
        node_features = node_features.contiguous()

        # Aggregate atomic energies
        image_idx = torch.arange(num_atoms.shape[0], device=edge_indices.device)
        image_idx = torch.repeat_interleave(image_idx, num_atoms)
        
        energy = torch.zeros(num_atoms.shape[0], device=num_atoms.device, dtype=torch.float32)
        energy.index_add_(0, image_idx, node_features)
        
        # Ensure energy is contiguous
        energy = energy.contiguous()

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
            inputs_list = torch.jit.annotate(List[Tensor], [edge_vectors.contiguous()])
            grad_outputs_list = torch.jit.annotate(Optional[List[Optional[Tensor]]], [torch.ones_like(energy)])
            
            dE_ddiff = torch.autograd.grad(
                outputs=outputs_list,
                inputs=inputs_list,
                grad_outputs=grad_outputs_list,
                retain_graph=self.training,
                create_graph=self.training,
            )[0]
            
            # Ensure gradients are contiguous
            dE_ddiff = dE_ddiff.contiguous()
            
            # Initialize forces
            i_forces = torch.zeros(positions.shape[0], 3, device=positions.device, dtype=torch.float32)
            j_forces = torch.zeros(positions.shape[0], 3, device=positions.device, dtype=torch.float32)
            i_forces.index_add_(0, edge_indices[:, 0], dE_ddiff)
            j_forces.index_add_(0, edge_indices[:, 1], -dE_ddiff)
            forces = i_forces + j_forces
            
            # Ensure forces are contiguous
            forces = forces.contiguous()
            
            data = replace_properties(data, forces=forces)

        return data 