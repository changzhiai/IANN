import torch
from torch import nn
from typing import Dict, List, Optional, Union, Callable
import warnings
import logging
import math

from e3nn import o3
from e3nn.nn import FullyConnectedNet

from iann.data import AtomsData, replace_properties
from .nequip import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    BesselBasis,
    PolynomialCutoff,
    SphericalHarmonicEdgeAttrs,
    AtomwiseLinear,
    PerTypeScaleShift,
    AtomwiseReduce,
    GradientOutput,
    resolve_cuequivariance,
    scatter_add,
    tp_path_exists,
)

try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet
    _HAS_CUEQUIVARIANCE = True
except (ImportError, SyntaxError, Exception) as e:
    _HAS_CUEQUIVARIANCE = False


class TwoBodyScalarEmbedding(nn.Module):
    def __init__(self, num_scalar_features: int, edge_dist_dim: int, node_dim: int, env_w_numel: int):
        super().__init__()
        self.num_scalar_features = num_scalar_features
        self.env_w_numel = env_w_numel
        
        self.mlp = FullyConnectedNet(
            [node_dim * 2 + edge_dist_dim, num_scalar_features, num_scalar_features + env_w_numel],
            torch.nn.functional.silu
        )

    def forward(self, data: AtomsData) -> tuple[AtomsData, dict]:
        node_feat = data.node_feat
        edge_dist_embedding = data.edge_dist_embedding
        edge_indices = data.edge_indices
        
        # features for atom i and j
        feat_i = node_feat[edge_indices[:, 0]]
        feat_j = node_feat[edge_indices[:, 1]]
        
        twobody_scalar_embed = torch.cat([feat_i, feat_j, edge_dist_embedding], dim=-1)
        projection = self.mlp(twobody_scalar_embed)
        
        twobody_scalar_features = projection[:, :self.num_scalar_features]
        env_w = projection[:, self.num_scalar_features:]
        
        state = {
            "twobody_scalar_features": twobody_scalar_features,
            "env_w": env_w
        }
        return data, state


class MakeWeightedChannels(nn.Module):
    def __init__(self, irreps_in: o3.Irreps, num_tensor_features: int):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.num_tensor_features = num_tensor_features
        self.weight_numel = len(self.irreps_in) * num_tensor_features

    def forward(self, tensor_basis: torch.Tensor, env_w: torch.Tensor) -> torch.Tensor:
        out = []
        w_index = 0
        b_index = 0
        for mul, ir in self.irreps_in:
            assert mul == 1
            dim = ir.dim
            w = env_w[:, w_index : w_index + self.num_tensor_features].unsqueeze(-1)
            b = tensor_basis[:, b_index : b_index + dim].unsqueeze(1)
            out.append((w * b).reshape(-1, self.num_tensor_features * dim))
            w_index += self.num_tensor_features
            b_index += dim
        return torch.cat(out, dim=-1)


class AllegroInteraction(nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        tensor_basis_irreps: o3.Irreps,
        num_scalar_features: int,
        num_tensor_features: int,
        tensor_track_allowed_irreps: o3.Irreps,
        num_accumulated_scalars: int,
        is_last_layer: bool,
        use_cue: bool = False,
    ):
        super().__init__()
        self.use_cue = use_cue
        self.num_scalar_features = num_scalar_features
        self.num_tensor_features = num_tensor_features
        self.is_last_layer = is_last_layer
        
        self.irreps_in = o3.Irreps(irreps_in)
        self.tensor_basis_irreps = o3.Irreps(tensor_basis_irreps)
        
        env_embed_irreps = o3.Irreps([(1, ir) for _, ir in self.tensor_basis_irreps])
        self.env_weighter = MakeWeightedChannels(
            irreps_in=self.tensor_basis_irreps,
            num_tensor_features=num_tensor_features
        )
        
        if is_last_layer:
            ir_out = o3.Irreps([(1, (0, 1))])
        else:
            ir_out = o3.Irreps(tensor_track_allowed_irreps)
        
        ir_out = o3.Irreps([
            (mul, ir) for mul, ir in ir_out
            if tp_path_exists(self.irreps_in, env_embed_irreps, ir)
        ])
        
        irreps_in1 = o3.Irreps([(num_tensor_features, ir) for _, ir in self.irreps_in])
        irreps_in2 = o3.Irreps([(num_tensor_features, ir) for _, ir in env_embed_irreps])
        irreps_out_tp = o3.Irreps([(num_tensor_features, ir) for _, ir in ir_out])
        
        if self.use_cue:
            self.tp = cuet.ChannelWiseTensorProduct(
                irreps_in1=cue.Irreps(cue.O3, irreps_in1),
                irreps_in2=cue.Irreps(cue.O3, irreps_in2),
                filter_irreps_out=cue.Irreps(cue.O3, irreps_out_tp),
                layout=cue.ir_mul,
                internal_weights=False,
                shared_weights=False,
            )
        else:
            instructions = []
            for i, (mul1, ir1) in enumerate(self.irreps_in):
                for j, (mul2, ir2) in enumerate(env_embed_irreps):
                    for k, (mul3, ir3) in enumerate(ir_out):
                        if ir3 in ir1 * ir2:
                            instructions.append((i, j, k, 'uvu', False))
                            
            self.tp = o3.TensorProduct(
                irreps_in1,
                irreps_in2,
                irreps_out_tp,
                instructions,
                shared_weights=False,
                internal_weights=False,
            )
            
        self.irreps_out = ir_out
        
        n_scalar_outs = 1
        self._n_scalar_outs = n_scalar_outs
        
        input_dim = num_accumulated_scalars + num_tensor_features * n_scalar_outs
        output_dim = num_scalar_features
        if not is_last_layer:
            output_dim += self.env_weighter.weight_numel
            
        self.latent = FullyConnectedNet(
            [input_dim, num_scalar_features, output_dim],
            torch.nn.functional.silu
        )

    def forward(self, data: AtomsData, state: dict) -> tuple[AtomsData, dict]:
        tensor_features = state["tensor_features"]
        accumulated_scalars = state["accumulated_scalars"]
        env_w = state["env_w"]
        
        tensor_basis = data.edge_diff_embedding
        edge_indices = data.edge_indices
        num_atoms = int(torch.sum(data.num_atoms))
        
        avg_num_neigh = getattr(self, "avg_num_neighbors", torch.tensor(1.0, device=tensor_basis.device))
        
        env_w_edges = self.env_weighter(tensor_basis, env_w)
        
        env_w_scatter = scatter_add(env_w_edges, edge_indices[:, 0], dim_size=num_atoms, dim=0)
        env_w_scatter = env_w_scatter / (avg_num_neigh ** 0.5)
        
        irin2 = env_w_scatter[edge_indices[:, 0]]
        
        tensor_features_out = self.tp(tensor_features, irin2)
        
        scalars = tensor_features_out[:, :self.num_tensor_features * self._n_scalar_outs]
        
        latents_input = torch.cat(accumulated_scalars + [scalars], dim=-1)
        latents_out = self.latent(latents_input)
        
        new_scalar_features = latents_out[:, :self.num_scalar_features]
        if not self.is_last_layer:
            new_env_w = latents_out[:, self.num_scalar_features:]
        else:
            new_env_w = None
            
        accumulated_scalars = accumulated_scalars + [new_scalar_features]
        
        state["tensor_features"] = tensor_features_out
        state["accumulated_scalars"] = accumulated_scalars
        state["env_w"] = new_env_w
        
        return data, state

    def datamodule(self, _datamodule):
        avg_num_neigh = _datamodule._get_avg_num_neighbors()
        if avg_num_neigh is not None:
            self.avg_num_neighbors = torch.tensor([avg_num_neigh])


class Allegro(torch.nn.Module):
    """
    A class to set up the Allegro model.
    """
    def __init__(
        self,
        num_layers: int,
        num_channels: int = 64,  # Used for node chemical embeddings
        num_scalar_features: int = 64, # Default scalar feature size for Allegro
        num_tensor_features: int = 16, # Default tensor feature size for Allegro
        norm_data: bool = False,
        norm_per_atom: bool = False,
        data_stddev: float = 1.0,
        data_mean: float = 0.0,
        per_type_energy_shifts: Optional[Union[float, List[float], Dict[str, float]]] = None,
        per_type_energy_scales: Optional[Union[float, List[float], Dict[str, float]]] = None,
        per_type_shifts_trainable: bool = False,
        per_type_scales_trainable: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.cutoff: float = kwargs.get('cutoff', 5.5)
        self.edge_sh_irreps: Union[o3.Irreps, str, None] = kwargs.get('edge_sh_irreps', None)
        self.tensor_track_allowed_irreps: Union[o3.Irreps, str, None] = kwargs.get('tensor_track_allowed_irreps', None)
        self.lmax: int = kwargs.get('lmax', 2)
        self.parity: bool = kwargs.get('parity', True)
        self.num_basis: int = kwargs.get('num_basis', 8)
        self.power: int = kwargs.get('power', 6)
        self.use_cue: bool = resolve_cuequivariance(kwargs.get('use_cue', None))
        
        species: List[str] = kwargs.get('species', None)
        self.universal_elems = kwargs.get('universal_elems', True)
        
        if self.universal_elems:
            num_elements = 119
        else:
            num_elements = len(species) if species else 119
        num_types = len(species) if species else 119
        
        if self.edge_sh_irreps is None:
            self.edge_sh_irreps = o3.Irreps.spherical_harmonics(self.lmax, p=-1 if self.parity else 1)
        else:
            self.edge_sh_irreps = o3.Irreps(self.edge_sh_irreps)
            
        if self.tensor_track_allowed_irreps is None:
            self.tensor_track_allowed_irreps = o3.Irreps(
                [
                    (1, (l, p))
                    for p in ((1, -1) if self.parity else (1,))
                    for l in range(self.lmax + 1)
                ]
            )
        else:
            self.tensor_track_allowed_irreps = o3.Irreps(self.tensor_track_allowed_irreps)

        # 1. Embeddings
        self.embeddings = nn.ModuleDict()
        self.embeddings['onehot_embedding'] = OneHotAtomEncoding(
            num_elements=num_elements, 
            species=species,
            universal_elems=self.universal_elems
        )
        self.embeddings['radial_basis'] = RadialBasisEdgeEncoding(
            basis=BesselBasis(cutoff=self.cutoff, num_basis=self.num_basis),
            cutoff_fn=PolynomialCutoff(cutoff=self.cutoff, power=self.power),
        )
        self.embeddings['sphere_harmonics'] = SphericalHarmonicEdgeAttrs(
            edge_sh_irreps=self.edge_sh_irreps
        )
        self.embeddings['chemical_embedding'] = AtomwiseLinear(
            irreps_in=self.embeddings.onehot_embedding.irreps_out['node_attr'],
            irreps_out=o3.Irreps([(num_channels, (0, 1))]),
            use_cue=self.use_cue,
        )
        
        # 2. Two Body Initial Embedding
        # Calculates env_w size for first layer
        first_weighter = MakeWeightedChannels(
            irreps_in=self.edge_sh_irreps,
            num_tensor_features=num_tensor_features
        )
        
        self.twobody_embedding = TwoBodyScalarEmbedding(
            num_scalar_features=num_scalar_features,
            edge_dist_dim=self.num_basis,
            node_dim=num_channels,
            env_w_numel=first_weighter.weight_numel
        )
        
        # Initial tensor features generator (repeating SH for tensor features)
        self.initial_tensor_features_weighter = MakeWeightedChannels(
            irreps_in=self.edge_sh_irreps,
            num_tensor_features=num_tensor_features
        )
        
        # 3. Allegro Interactions
        self.interactions = nn.ModuleList()
        current_irreps = self.edge_sh_irreps
        for i in range(num_layers):
            is_last_layer = (i == num_layers - 1)
            num_accumulated_scalars = num_scalar_features * (i + 1)
            
            interaction = AllegroInteraction(
                irreps_in=current_irreps,
                tensor_basis_irreps=self.edge_sh_irreps,
                num_scalar_features=num_scalar_features,
                num_tensor_features=num_tensor_features,
                tensor_track_allowed_irreps=self.tensor_track_allowed_irreps,
                num_accumulated_scalars=num_accumulated_scalars,
                is_last_layer=is_last_layer,
                use_cue=self.use_cue,
            )
            self.interactions.append(interaction)
            current_irreps = interaction.irreps_out
            
        # 4. Edge Readout MLP
        self.edge_readout_mlp = FullyConnectedNet(
            [num_scalar_features * (num_layers + 1), max(1, num_scalar_features // 2), 1],
            torch.nn.functional.silu
        )
        
        # Normalization (global)
        self.energy_bias = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.norm_data = torch.nn.Parameter(torch.tensor(norm_data), requires_grad=False)
        self.norm_per_atom = torch.nn.Parameter(torch.tensor(norm_per_atom), requires_grad=False)
        self.data_stddev = torch.nn.Parameter(torch.tensor(data_stddev), requires_grad=False)
        self.data_mean = torch.nn.Parameter(torch.tensor(data_mean), requires_grad=False)

        # Per-type energy scale and shift
        self.use_per_type_scale_shift = (
            per_type_energy_shifts is not None or per_type_energy_scales is not None
        )
        self.per_type_scale_shift: Optional[PerTypeScaleShift] = None
        if self.use_per_type_scale_shift:
            self.per_type_scale_shift = PerTypeScaleShift(
                num_types=119,
                shifts=per_type_energy_shifts,
                scales=per_type_energy_scales,
                shifts_trainable=per_type_shifts_trainable,
                scales_trainable=per_type_scales_trainable,
                species=species,
            )
            
        self.atomwise_reduce = AtomwiseReduce(output_key='energy')
        
        self.compute_forces = kwargs.get('compute_forces', False)
        self.compute_stress = kwargs.get('compute_stress', False)
        self.compute_virial = kwargs.get('compute_virial', False)
        
        outputs = []
        if self.compute_forces: outputs.append('forces')
        if self.compute_stress: outputs.append('stress')
        if self.compute_virial: outputs.append('virial')
        
        if outputs:
            self.gradient_output = GradientOutput(model_outputs=outputs)
        else:
            self.gradient_output = None

    def _apply_displacement(self, data: AtomsData) -> AtomsData:
        image_indices = data.image_indices
        assert image_indices is not None, "No image indices found!"
        num_images = int(image_indices.max() + 1)
        displacement = torch.zeros(
            (num_images, 3, 3), 
            dtype=data.edge_vectors.dtype, 
            device=data.edge_vectors.device, 
        ).requires_grad_()
        image_idx = image_indices[data.edge_indices[:, 0]]
        disp_batch = displacement[image_idx]
        edge_vectors = data.edge_vectors + torch.bmm(
            disp_batch, data.edge_vectors.unsqueeze(-1)
        ).squeeze(-1)
        
        return replace_properties(data, edge_vectors=edge_vectors, displacement=displacement)

    def _forward_network(self, data: AtomsData) -> AtomsData:
        for m in self.embeddings.values():
            data = m(data)
            
        data, state = self.twobody_embedding(data)
        state["accumulated_scalars"] = [state["twobody_scalar_features"]]
        
        # Construct initial tensor features using a fixed ones weight
        # This acts as an unweighted embedding into num_tensor_features channels
        ones_w = torch.ones(
            (data.edge_diff_embedding.size(0), self.initial_tensor_features_weighter.weight_numel),
            dtype=data.edge_diff_embedding.dtype,
            device=data.edge_diff_embedding.device
        )
        initial_tensor_features = self.initial_tensor_features_weighter(data.edge_diff_embedding, ones_w)
        state["tensor_features"] = initial_tensor_features
        
        for m in self.interactions:
            data, state = m(data, state)
            
        final_scalars = torch.cat(state["accumulated_scalars"], dim=-1)
        edge_energy = self.edge_readout_mlp(final_scalars).squeeze(-1)
        
        # Scatter edge energies to atomic energies
        num_atoms = int(torch.sum(data.num_atoms))
        atomic_energy = scatter_add(edge_energy, data.edge_indices[:, 0], dim_size=num_atoms, dim=0)
        atomic_energy = atomic_energy + self.energy_bias
        
        return replace_properties(data, atomic_energy=atomic_energy)

    def _apply_scale_shift(self, data: AtomsData) -> AtomsData:
        atomic_energy = data.atomic_energy
        if self.per_type_scale_shift is not None:
            atomic_energy = self.per_type_scale_shift(atomic_energy, data.atomic_numbers.long())

        data = replace_properties(data, atomic_energy=atomic_energy)
        data = self.atomwise_reduce(data)

        if not self.use_per_type_scale_shift and self.norm_data:
            energy = data.energy
            if energy is not None:
                normalizer = self.data_stddev
                energy = normalizer * energy
                if self.norm_per_atom:
                    mean_shift = data.num_atoms.to(energy.dtype) * self.data_mean
                else:
                    mean_shift = self.data_mean
                    
                energy = energy + mean_shift
                data = replace_properties(data, energy=energy)
            
        return data

    def forward(self, data: AtomsData) -> AtomsData:
        """
        Parameters
        ----------
        data : AtomsData
            Input data for the model.

        Returns
        -------
        AtomsData
            Output data after applying the model.
        """
        # 1. Pre-process for Stress/Virial
        if self.compute_stress or self.compute_virial:
            data = self._apply_displacement(data)
            
        # 2. Core Neural Network
        data = self._forward_network(data)
        
        # 3. Post-process (Scaling & Reduction)
        data = self._apply_scale_shift(data)
        
        # 4. Gradients (Forces, Stress, Virial)
        if self.gradient_output is not None:
            data = self.gradient_output(data)
            
        return data

    def datamodule(self, _datamodule):
        for interaction in self.interactions:
            if hasattr(interaction, 'datamodule'):
                interaction.datamodule(_datamodule)
    
    def get_optimization_info(self):
        return {
            "cuequivariance_available": _HAS_CUEQUIVARIANCE,
            "optimization_enabled": self.use_cue,
            "performance_boost": "2-5x speedup" if self.use_cue else "No optimization"
        }
