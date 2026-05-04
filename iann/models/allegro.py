"""Allegro model.

A single-file, math-faithful port of the upstream `mir-group/allegro`
architecture (https://github.com/mir-group/allegro), reusing the IANN
data/training conventions and helper modules already exposed by
``iann/models/nequip.py``.

Key components, in pipeline order:

1. ``OneHotAtomEncoding`` (from nequip.py): one-hot atom encoding + TypeMapper.
2. ``RadialBasisEdgeEncoding`` (from nequip.py): Bessel basis * polynomial cutoff,
   produces per-edge radial scalar features.
3. ``SphericalHarmonicEdgeAttrs`` (from nequip.py): spherical harmonics of edge
   vectors, produces per-edge tensor basis (multiplicity 1 per irrep).
4. ``ProductTypeEmbedding``: per-edge ``(center_embed, neighbor_embed)`` concat
   *element-wise multiplied* by a linear projection of the radial basis. This
   replaces the simplified concat-MLP two-body embedding of the previous file.
5. ``ScalarMLP`` (scalar_embed_mlp): refines the two-body scalar embedding.
6. ``TwoBodySphericalHarmonicTensorEmbed``: learned weighted spherical harmonics
   = initial per-edge tensor features (replaces the previous file's all-ones
   weighting).
7. ``AllegroModule``: the main interaction stack. Per layer it computes a
   per-node environment by scattering env-weighted spherical harmonics, pulls
   it back per-edge, feeds it through an ``o3.TensorProduct`` ('uuu'
   instructions) against the running tensor features, extracts the scalar
   subspace, and updates the DenseNet-style scalar track via a
   ``ScalarMLP`` that also emits the next layer's env weights.
8. ``ScalarMLP`` (edge_readout): maps the concatenated scalar track to a
   per-edge energy.
9. ``EdgewiseReduce``: scatter per-edge energies onto atoms with
   ``1/sqrt(avg_num_neighbors)`` and ``1/sqrt(2)`` normalization (the latter
   accounts for ``dE/dr_i`` summing both ``dE/dr_ij`` and ``dE/dr_ji``).
10. ``PerTypeScaleShift`` + ``AtomwiseReduce`` + ``GradientOutput``: per-type
    energy shift/scale, atom -> total energy reduction, and autograd
    forces / stresses / virials. All reused from ``iann/models/nequip.py``.

The tensor product is implemented via ``e3nn.o3.TensorProduct`` with 'uuu'
instructions and trainable per-path scalar weights. This matches the
``tp_path_channel_coupling=False`` math of the upstream strided
``Contracter``. The cuEquivariance ``ChannelWiseTensorProduct`` expects
``mul=1`` on its second input, which doesn't fit Allegro's symmetric
``(M, ...) (x) (M, ...) -> (M, ...)`` setup, so the ``use_cue`` flag is kept
for API parity but the Allegro TPs always run on e3nn.
"""

import math
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from e3nn import o3

from iann.data import AtomsData, replace_properties
from .nequip import (
    OneHotAtomEncoding,
    BesselBasis,
    PolynomialCutoff,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
    PerTypeScaleShift,
    AtomwiseReduce,
    GradientOutput,
    scatter_add,
    tp_path_exists,
    resolve_cuequivariance,
)

logging.getLogger("cuequivariance").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="The TorchScript type system doesn't support instance-level annotations")

try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet
    _HAS_CUEQUIVARIANCE = True
except (ImportError, SyntaxError, Exception):
    _HAS_CUEQUIVARIANCE = False


# ---------------------------------------------------------------------------
# 1. ScalarMLP
# ---------------------------------------------------------------------------

_NONLIN_MAP = {
    "silu": torch.nn.functional.silu,
    "mish": torch.nn.functional.mish,
    "gelu": torch.nn.functional.gelu,
}


class ScalarMLP(nn.Module):
    """Lightweight replacement for upstream ``ScalarMLP``/``ScalarMLPFunction``.

    Forward-init mode initializes weights so that the variance of activations
    is preserved through the (chosen) nonlinearity, which is what makes
    upstream Allegro train stably without explicit normalization layers.
    """

    is_nonlinear: bool

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers_depth: int = 0,
        hidden_layers_width: int = 64,
        nonlinearity: Optional[str] = "silu",
        bias: bool = False,
        forward_weight_init: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_nonlinear = (hidden_layers_depth > 0) and (nonlinearity is not None)

        if nonlinearity is None:
            self.act = None
        elif nonlinearity in _NONLIN_MAP:
            self.act = _NONLIN_MAP[nonlinearity]
        else:
            raise ValueError(f"Unknown nonlinearity {nonlinearity!r}")

        dims: List[int] = [input_dim]
        for _ in range(hidden_layers_depth):
            dims.append(hidden_layers_width)
        dims.append(output_dim)

        self.linears = nn.ModuleList()
        n_layers = len(dims) - 1
        for i in range(n_layers):
            linear = nn.Linear(dims[i], dims[i + 1], bias=bias)
            if forward_weight_init:
                fan_in = dims[i]
                # Last layer uses unit gain (linear output); preceding layers
                # account for the variance reduction of the chosen activation.
                if (i < n_layers - 1) and (self.act is not None):
                    # SiLU/Mish/GELU all have approximately gain ~ sqrt(2)
                    gain = math.sqrt(2.0)
                else:
                    gain = 1.0
                std = gain / math.sqrt(fan_in)
                nn.init.normal_(linear.weight, mean=0.0, std=std)
                if bias:
                    nn.init.zeros_(linear.bias)
            self.linears.append(linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = len(self.linears)
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if (i < n - 1) and (self.act is not None):
                x = self.act(x)
        return x


# ---------------------------------------------------------------------------
# 2. MakeWeightedChannels
# ---------------------------------------------------------------------------


class MakeWeightedChannels(nn.Module):
    """Weight an ``mul=1`` irrep tensor (e.g. spherical harmonics) to ``M``
    output channels per irrep.

    Port of ``allegro/nn/_strided/_channels.py``. The output is flattened in
    e3nn's standard ``mul_ir`` layout: for output irreps
    ``(M, l1) + (M, l2) + ...`` the flat tensor has, per irrep block,
    channel-outer / dim-inner ordering.

    Parameters
    ----------
    irreps_in : o3.Irreps
        Input irreps; every multiplicity must be 1.
    multiplicity_out : int
        Number of output channels ``M``.
    alpha : float
        Optional scalar normalization absorbed into the weighting.
    weight_individual_irreps : bool
        If True, each (channel, irrep) pair gets its own scalar weight, giving
        ``weight_numel = num_irreps * M``. If False, each channel gets a
        single scalar weight that is shared across all irreps,
        ``weight_numel = M``.
    """

    weight_numel: int
    multiplicity_out: int
    weight_individual_irreps: bool
    alpha: float
    _num_irreps: int

    def __init__(
        self,
        irreps_in,
        multiplicity_out: int,
        alpha: float = 1.0,
        weight_individual_irreps: bool = True,
    ):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)
        assert all(mul == 1 for mul, _ in irreps_in), \
            "MakeWeightedChannels requires mul=1 input irreps"
        assert multiplicity_out >= 1
        self.irreps_in = irreps_in
        self._num_irreps = len(irreps_in)
        self.multiplicity_out = multiplicity_out
        self.weight_individual_irreps = weight_individual_irreps
        self.alpha = float(alpha)

        if weight_individual_irreps:
            self.weight_numel = self._num_irreps * multiplicity_out
        else:
            self.weight_numel = multiplicity_out

        # Per-irrep dim slices (used to walk the output in mul_ir layout).
        offsets = [0]
        for _, ir in irreps_in:
            offsets.append(offsets[-1] + ir.dim)
        # Stored as plain Python lists; small (<= ~10 entries).
        self._irrep_offsets: List[int] = offsets
        self._total_dim: int = offsets[-1]

    def forward(self, edge_attr: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        edge_attr: ``[num_edges, total_dim]`` (mul=1 input, so flat = sum 2l+1).
        weights:   ``[num_edges, weight_numel]``.

        Returns ``[num_edges, sum_i M*(2l_i+1)]`` in e3nn ``mul_ir`` layout.
        """
        num_edges = edge_attr.size(0)
        M = self.multiplicity_out
        offsets = self._irrep_offsets

        if self.weight_individual_irreps:
            # weights: [z, M*num_irreps] with order [ch1_ir1, ch1_ir2, ..., ch1_irK,
            #                                        ch2_ir1, ..., chM_irK]
            w_per = weights.view(num_edges, M, self._num_irreps)  # [z, M, K]
            chunks: List[torch.Tensor] = []
            for i in range(self._num_irreps):
                w_i = w_per[:, :, i].unsqueeze(-1)                    # [z, M, 1]
                b_i = self.alpha * edge_attr[:, offsets[i]:offsets[i + 1]].unsqueeze(1)  # [z, 1, dim]
                chunks.append((w_i * b_i).reshape(num_edges, -1))     # [z, M*dim]
            return torch.cat(chunks, dim=-1)
        else:
            # weights: [z, M], shared across irreps.
            w = weights.unsqueeze(-1)                                  # [z, M, 1]
            chunks2: List[torch.Tensor] = []
            for i in range(self._num_irreps):
                b_i = self.alpha * edge_attr[:, offsets[i]:offsets[i + 1]].unsqueeze(1)  # [z, 1, dim]
                chunks2.append((w * b_i).reshape(num_edges, -1))
            return torch.cat(chunks2, dim=-1)


# ---------------------------------------------------------------------------
# 3. ProductTypeEmbedding
# ---------------------------------------------------------------------------


class ProductTypeEmbedding(nn.Module):
    """Two-body scalar embedding via element-wise product of
    ``(center_embed, neighbor_embed)`` (concatenated to ``dim``) and
    ``Linear(radial_basis)``.

    Port of ``allegro/nn/_edgeembed.py::ProductTypeEmbedding``.
    """

    def __init__(
        self,
        num_types: int,
        initial_embedding_dim: int,
        radial_dim: int,
        forward_weight_init: bool = True,
    ):
        super().__init__()
        assert initial_embedding_dim % 2 == 0, \
            "`initial_embedding_dim` must be an even number"
        self.num_types = num_types
        self.initial_embedding_dim = initial_embedding_dim

        self.center_embed = nn.Embedding(num_types, initial_embedding_dim // 2)
        self.neighbor_embed = nn.Embedding(num_types, initial_embedding_dim // 2)

        # Linear projection of radial basis -> initial_embedding_dim
        self.basis_linear = ScalarMLP(
            input_dim=radial_dim,
            output_dim=initial_embedding_dim,
            hidden_layers_depth=0,
            nonlinearity=None,
            bias=False,
            forward_weight_init=forward_weight_init,
        )
        assert not self.basis_linear.is_nonlinear

    def forward(self, data: AtomsData, edge_types: torch.Tensor) -> AtomsData:
        # edge_types: [num_edges, 2] - (center_type, neighbor_type) per edge.
        center_t = self.center_embed(edge_types[:, 0])
        neighbor_t = self.neighbor_embed(edge_types[:, 1])
        type_embed = torch.cat([center_t, neighbor_t], dim=-1)

        radial = data.edge_dist_embedding
        assert radial is not None, "edge_dist_embedding must be populated by RadialBasisEdgeEncoding"
        basis = self.basis_linear(radial)
        out = type_embed * basis

        return replace_properties(data, edge_dist_embedding=out)


# ---------------------------------------------------------------------------
# 4. TwoBodySphericalHarmonicTensorEmbed
# ---------------------------------------------------------------------------


class TwoBodySphericalHarmonicTensorEmbed(nn.Module):
    """Construct two-body tensor embedding as learned-weighted spherical
    harmonics.

    Port of ``allegro/nn/tensorembed.py``. Reads the spherical-harmonic
    ``edge_diff_embedding`` written by ``SphericalHarmonicEdgeAttrs`` (so the
    SH itself is computed once and shared with ``AllegroModule`` as the tensor
    basis), and produces initial per-edge tensor features by weighting it with
    a linear projection of the scalar embedding.
    """

    def __init__(
        self,
        edge_sh_irreps,
        scalar_embed_dim: int,
        num_tensor_features: int,
        weight_individual_irreps: bool = True,
        forward_weight_init: bool = True,
    ):
        super().__init__()
        self.edge_sh_irreps = o3.Irreps(edge_sh_irreps)
        self.num_tensor_features = num_tensor_features

        self._edge_weighter = MakeWeightedChannels(
            irreps_in=self.edge_sh_irreps,
            multiplicity_out=num_tensor_features,
            weight_individual_irreps=weight_individual_irreps,
        )

        self.env_embed_linear = ScalarMLP(
            input_dim=scalar_embed_dim,
            output_dim=self._edge_weighter.weight_numel,
            hidden_layers_depth=0,
            nonlinearity=None,
            bias=False,
            forward_weight_init=forward_weight_init,
        )
        assert not self.env_embed_linear.is_nonlinear

        self.irreps_out = o3.Irreps(
            [(num_tensor_features, ir) for _, ir in self.edge_sh_irreps]
        )

    def forward(self, data: AtomsData) -> Tuple[AtomsData, torch.Tensor]:
        edge_sh = data.edge_diff_embedding
        scalar_embed = data.edge_dist_embedding
        assert edge_sh is not None and scalar_embed is not None
        weights = self.env_embed_linear(scalar_embed)
        tensor_features = self._edge_weighter(edge_sh, weights)
        return data, tensor_features


# ---------------------------------------------------------------------------
# 5. AllegroModule
# ---------------------------------------------------------------------------


class AllegroModule(nn.Module):
    """Stack of Allegro interaction layers.

    Port of ``allegro/nn/_allegro.py::Allegro_Module``.

    Each layer:

    1. Weights the (mul=1) tensor basis with per-edge ``env_w`` -> weighted
       per-edge tensor of multiplicity ``M`` (``MakeWeightedChannels``).
    2. Scatters the weighted tensor onto its center node, divides by
       ``sqrt(avg_num_neighbors)``.
    3. Pulls back the per-node env to per-edge.
    4. Tensor-products the running per-edge tensor features against this
       env tensor (e3nn 'uuu' instructions, internal trainable per-path
       weights -> ``tp_path_channel_coupling=False`` math).
    5. Extracts the scalar (l=0) subspace of the TP output.
    6. DenseNet-style: concats all previously accumulated scalar features
       with the new scalars and feeds them through a ``ScalarMLP`` (latent),
       which outputs the next layer's scalar features and (except at the
       last layer) the next layer's ``env_w`` weights.
    """

    num_layers: int
    num_scalar_features: int
    num_tensor_features: int

    def __init__(
        self,
        num_layers: int,
        num_scalar_features: int,
        num_tensor_features: int,
        scalar_input_dim: int,
        tensor_basis_irreps,
        tensor_track_allowed_irreps,
        avg_num_neighbors: Optional[float] = None,
        weight_individual_irreps: bool = True,
        latent_hidden_layers_depth: int = 1,
        latent_hidden_layers_width: int = 64,
        latent_nonlinearity: Optional[str] = "silu",
        forward_weight_init: bool = True,
        use_cue: bool = False,
    ):
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        self.num_scalar_features = num_scalar_features
        self.num_tensor_features = num_tensor_features
        self.use_cue = use_cue

        self.tensor_basis_irreps = o3.Irreps(tensor_basis_irreps)
        self.tensor_track_allowed_irreps = o3.Irreps(tensor_track_allowed_irreps)
        assert all(mul == 1 for mul, _ in self.tensor_track_allowed_irreps), \
            "tensor_track_allowed_irreps must have multiplicity 1"

        # Env weighter (used in every layer to weight the SH tensor basis).
        self._env_weighter = MakeWeightedChannels(
            irreps_in=self.tensor_basis_irreps,
            multiplicity_out=num_tensor_features,
            weight_individual_irreps=weight_individual_irreps,
        )
        env_embed_irreps = o3.Irreps([(1, ir) for _, ir in self.tensor_basis_irreps])
        SCALAR = o3.Irrep(0, 1)
        assert env_embed_irreps[0].ir == SCALAR, \
            "env_embed_irreps must start with a scalar"

        # First-layer linear: two-body scalar embedding ->
        # (twobody_scalar_features, env_w for first TP).
        self.first_layer_env_embed_projection = ScalarMLP(
            input_dim=scalar_input_dim,
            output_dim=num_scalar_features + self._env_weighter.weight_numel,
            hidden_layers_depth=0,
            nonlinearity=None,
            bias=False,
            forward_weight_init=forward_weight_init,
        )
        assert not self.first_layer_env_embed_projection.is_nonlinear

        # ---- Build per-layer irrep tracks (forward then backward pruning) ----
        arg_irreps = env_embed_irreps   # initial tensor features have these irreps (with mul=M)
        tps_irreps_seq: List[o3.Irreps] = [arg_irreps]
        for layer_idx in range(num_layers):
            if layer_idx == num_layers - 1:
                ir_out = o3.Irreps([(1, (0, 1))])
            else:
                ir_out = self.tensor_track_allowed_irreps
            ir_out = o3.Irreps([
                (mul, ir) for mul, ir in ir_out
                if tp_path_exists(arg_irreps, env_embed_irreps, ir)
            ])
            arg_irreps = ir_out
            tps_irreps_seq.append(ir_out)

        # Backward pruning: remove arg irreps whose products with env can never
        # reach the final scalar.
        out_irreps_back = tps_irreps_seq[-1]
        new_seq: List[o3.Irreps] = [out_irreps_back]
        for arg_irreps_b in reversed(tps_irreps_seq[:-1]):
            new_args: List[Tuple[int, o3.Irrep]] = []
            for mul, arg_ir in arg_irreps_b:
                useful = False
                for _, env_ir in env_embed_irreps:
                    if any(i in out_irreps_back for i in arg_ir * env_ir):
                        useful = True
                        break
                if useful:
                    new_args.append((mul, arg_ir))
            new_irreps = o3.Irreps(new_args)
            new_seq.append(new_irreps)
            out_irreps_back = new_irreps
        tps_irreps_seq = list(reversed(new_seq))
        assert tps_irreps_seq[-1].lmax == 0, "last layer must produce scalars only"

        tps_irreps_in = tps_irreps_seq[:-1]
        tps_irreps_out = tps_irreps_seq[1:]

        # ---- Build TPs and latent MLPs ----
        self.tps = nn.ModuleList()
        self.latents = nn.ModuleList()
        self._n_scalar_outs: List[int] = []
        # Stored slice sizes for fast scalar extraction in forward.
        self._n_scalar_slice: List[int] = []

        for layer_idx, (arg_irreps_layer, out_irreps_layer) in enumerate(
            zip(tps_irreps_in, tps_irreps_out)
        ):
            irin1 = o3.Irreps([(num_tensor_features, ir) for _, ir in arg_irreps_layer])
            irin2 = o3.Irreps([(num_tensor_features, ir) for _, ir in env_embed_irreps])
            irout = o3.Irreps([(num_tensor_features, ir) for _, ir in out_irreps_layer])

            instructions = []
            for i_in, (_, ir1) in enumerate(irin1):
                for i_env, (_, ir2) in enumerate(irin2):
                    for i_out, (_, ir3) in enumerate(irout):
                        if ir3 in ir1 * ir2:
                            # 'uuu' + train=True => trainable per-path scalar weight,
                            # shared across batch and channels.
                            instructions.append((i_in, i_env, i_out, "uuu", True))

            tp = o3.TensorProduct(
                irin1,
                irin2,
                irout,
                instructions,
                shared_weights=True,
                internal_weights=True,
            )
            self.tps.append(tp)

            # Number of scalar irreps at the start of out_irreps_layer.
            n_scalar_outs = 0
            for _, ir in out_irreps_layer:
                if ir == SCALAR:
                    n_scalar_outs += 1
                else:
                    break
            assert n_scalar_outs >= 1, \
                f"out_irreps must start with at least one scalar, got {out_irreps_layer}"
            self._n_scalar_outs.append(n_scalar_outs)
            self._n_scalar_slice.append(num_tensor_features * n_scalar_outs)

            # Latent MLP: concat(all previous scalar features, this layer's
            # extracted scalars) -> (next scalar features [, next env_w]).
            latent_input_dim = (
                num_scalar_features * (layer_idx + 1)
                + num_tensor_features * n_scalar_outs
            )
            latent_output_dim = num_scalar_features
            if layer_idx < num_layers - 1:
                latent_output_dim += self._env_weighter.weight_numel
            self.latents.append(ScalarMLP(
                input_dim=latent_input_dim,
                output_dim=latent_output_dim,
                hidden_layers_depth=latent_hidden_layers_depth,
                hidden_layers_width=latent_hidden_layers_width,
                nonlinearity=latent_nonlinearity,
                bias=False,
                forward_weight_init=forward_weight_init,
            ))

        self.scalar_out_dim = num_scalar_features * (num_layers + 1)

        # avg_num_neighbors buffer (settable via datamodule(...) hook).
        if avg_num_neighbors is not None:
            self._initialized = True
            avg_n = torch.tensor([float(avg_num_neighbors)])
        else:
            self._initialized = False
            avg_n = torch.ones(1)
        self.register_buffer("avg_num_neighbors", avg_n)

    def forward(
        self,
        data: AtomsData,
        twobody_scalars: torch.Tensor,
        tensor_features: torch.Tensor,
    ) -> Tuple[AtomsData, torch.Tensor]:
        edge_indices = data.edge_indices
        edge_center = edge_indices[:, 0]
        num_atoms = int(torch.sum(data.num_atoms))
        tensor_basis = data.edge_diff_embedding
        assert tensor_basis is not None

        # First-layer linear projection: two-body scalars -> twobody features + env_w.
        projection = self.first_layer_env_embed_projection(twobody_scalars)
        twobody_scalar_features = projection[:, :self.num_scalar_features]
        accumulated: List[torch.Tensor] = [twobody_scalar_features]
        env_w = projection[:, self.num_scalar_features:]

        for layer_idx in range(self.num_layers):
            tp = self.tps[layer_idx]
            latent = self.latents[layer_idx]

            # 1. weight tensor basis with env_w
            env_w_edges = self._env_weighter(tensor_basis, env_w)
            # 2. scatter to nodes (per-atom env feature)
            node_env = scatter_add(env_w_edges, edge_center, dim_size=num_atoms, dim=0)
            # 3. normalize by sqrt(avg_num_neighbors)
            node_env = node_env / (self.avg_num_neighbors ** 0.5)
            # 4. pull back per-edge as TP input 2
            irin2 = node_env[edge_center]
            # 5. tensor product
            tensor_features = tp(tensor_features, irin2)
            # 6. extract scalar subspace (channel-outer / dim-inner; scalars are
            #    the first n_scalar_outs irreps so a contiguous slice works).
            n_slice = self._n_scalar_slice[layer_idx]
            scalars = tensor_features[:, :n_slice]
            # 7. latent MLP on densenet-concatenated scalar track.
            latent_input = torch.cat(accumulated + [scalars], dim=-1)
            latent_output = latent(latent_input)
            # 8. split into (new scalar features, new env_w).
            new_scalar_features = latent_output[:, :self.num_scalar_features]
            accumulated.append(new_scalar_features)
            if layer_idx < self.num_layers - 1:
                env_w = latent_output[:, self.num_scalar_features:]

        final_scalars = torch.cat(accumulated, dim=-1)
        return data, final_scalars

    def datamodule(self, _datamodule):
        if not self._initialized:
            avg_num_neigh = _datamodule._get_avg_num_neighbors()
            if avg_num_neigh is not None:
                self.avg_num_neighbors = torch.tensor(
                    [float(avg_num_neigh)],
                    device=self.avg_num_neighbors.device,
                    dtype=self.avg_num_neighbors.dtype,
                )
                self._initialized = True


# ---------------------------------------------------------------------------
# 6. EdgewiseReduce
# ---------------------------------------------------------------------------


class EdgewiseReduce(nn.Module):
    """Sum per-edge scalars onto centers, with the upstream-Allegro
    normalization: divide by ``sqrt(avg_num_neighbors)`` and ``sqrt(2)``.

    The ``sqrt(2)`` factor accounts for ``dE/dr_i`` collecting contributions
    from both ``dE/dr_ij`` and ``dE/dr_ji`` when the dataloader emits both
    directions of every edge (which IANN's ``AseDataReader`` does).
    """

    def __init__(self, avg_num_neighbors: Optional[float] = None):
        super().__init__()
        if avg_num_neighbors is not None:
            self._initialized = True
            avg_n = torch.tensor([float(avg_num_neighbors)])
        else:
            self._initialized = False
            avg_n = torch.ones(1)
        self.register_buffer("avg_num_neighbors", avg_n)
        self._sqrt2 = math.sqrt(2.0)

    def forward(
        self,
        edge_data: torch.Tensor,
        edge_indices: torch.Tensor,
        num_atoms: int,
    ) -> torch.Tensor:
        edge_center = edge_indices[:, 0]
        out = scatter_add(edge_data, edge_center, dim_size=num_atoms, dim=0)
        out = out / (self.avg_num_neighbors ** 0.5)
        out = out / self._sqrt2
        return out

    def datamodule(self, _datamodule):
        if not self._initialized:
            avg_num_neigh = _datamodule._get_avg_num_neighbors()
            if avg_num_neigh is not None:
                self.avg_num_neighbors = torch.tensor(
                    [float(avg_num_neigh)],
                    device=self.avg_num_neighbors.device,
                    dtype=self.avg_num_neighbors.dtype,
                )
                self._initialized = True


# ---------------------------------------------------------------------------
# 7. Allegro top-level model
# ---------------------------------------------------------------------------


class Allegro(nn.Module):
    """Allegro: strictly local equivariant interatomic potential.

    Faithful (math-equivalent) port of
    https://github.com/mir-group/allegro/tree/main/allegro into a single file
    that fits the IANN ``AtomsData`` / ``Trainer`` conventions.

    Notable kwargs (a superset of what ``test/allegro/allegro.py`` passes):

    - ``num_layers``: number of Allegro interaction layers.
    - ``num_channels``: dimension of ``ProductTypeEmbedding`` (must be even).
      Kept for backward-compat with the existing IANN trainer config.
    - ``num_scalar_features``: width of the DenseNet scalar track.
    - ``num_tensor_features``: multiplicity of the equivariant tensor track.
    - ``lmax``, ``parity``: spherical-harmonic / tensor-track irreps.
    - ``cutoff``, ``num_basis``, ``power``: radial basis / cutoff settings.
    - ``species`` / ``universal_elems``: chemistry encoding.
    - ``per_type_energy_*`` / ``norm_*`` / ``data_*``: energy normalization.
    - ``compute_forces`` / ``compute_stress`` / ``compute_virial``: which
      gradient outputs to produce.
    - ``use_cue``: kept for API parity (cuEq's ChannelWiseTensorProduct does
      not fit the symmetric Allegro TP, so this flag does not actually change
      the TP backend in this file).
    """

    def __init__(
        self,
        num_layers: int,
        num_channels: int = 64,
        num_scalar_features: int = 64,
        num_tensor_features: int = 16,
        norm_data: bool = False,
        norm_per_atom: bool = False,
        data_stddev: Union[float, List[float]] = 1.0,
        data_mean: Union[float, List[float]] = 0.0,
        per_type_energy_shifts: Optional[Union[float, List[float], Dict[str, float]]] = None,
        per_type_energy_scales: Optional[Union[float, List[float], Dict[str, float]]] = None,
        per_type_shifts_trainable: bool = False,
        per_type_scales_trainable: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.cutoff: float = kwargs.get("cutoff", 5.5)
        self.lmax: int = kwargs.get("lmax", 2)
        self.parity: bool = kwargs.get("parity", True)
        self.num_basis: int = kwargs.get("num_basis", 8)
        self.power: int = kwargs.get("power", 6)

        self.edge_sh_irreps: Union[o3.Irreps, str, None] = kwargs.get("edge_sh_irreps", None)
        self.tensor_track_allowed_irreps: Union[o3.Irreps, str, None] = kwargs.get(
            "tensor_track_allowed_irreps", None
        )

        self.use_cue: bool = resolve_cuequivariance(kwargs.get("use_cue", None))

        # Allegro-specific MLP / weighting hyperparameters.
        two_body_mlp_hidden_layers_depth = int(kwargs.get("two_body_mlp_hidden_layers_depth", 1))
        two_body_mlp_hidden_layers_width = int(
            kwargs.get("two_body_mlp_hidden_layers_width", num_scalar_features)
        )
        two_body_mlp_nonlinearity = kwargs.get("two_body_mlp_nonlinearity", "silu")
        allegro_mlp_hidden_layers_depth = int(kwargs.get("allegro_mlp_hidden_layers_depth", 1))
        allegro_mlp_hidden_layers_width = int(
            kwargs.get("allegro_mlp_hidden_layers_width", num_scalar_features)
        )
        allegro_mlp_nonlinearity = kwargs.get("allegro_mlp_nonlinearity", "silu")
        readout_mlp_hidden_layers_depth = int(kwargs.get("readout_mlp_hidden_layers_depth", 1))
        readout_mlp_hidden_layers_width = int(
            kwargs.get("readout_mlp_hidden_layers_width", max(1, num_scalar_features // 2))
        )
        readout_mlp_nonlinearity = kwargs.get("readout_mlp_nonlinearity", "silu")
        weight_individual_irreps: bool = bool(kwargs.get("weight_individual_irreps", True))
        forward_weight_init: bool = bool(kwargs.get("forward_weight_init", True))
        avg_num_neighbors_init: Optional[float] = kwargs.get("avg_num_neighbors", None)

        species: Optional[List[str]] = kwargs.get("species", None)
        self.universal_elems = bool(kwargs.get("universal_elems", True))
        self.species = species

        # Number of types used for the type embedding: 119 in universal mode,
        # otherwise len(species) (or 119 if species is missing).
        if self.universal_elems:
            num_types = 119
        else:
            num_types = len(species) if species else 119
        self.num_types = num_types

        # ----- Resolve irreps -----
        if self.edge_sh_irreps is None:
            self.edge_sh_irreps = o3.Irreps.spherical_harmonics(
                self.lmax, p=-1 if self.parity else 1
            )
        else:
            self.edge_sh_irreps = o3.Irreps(self.edge_sh_irreps)

        if self.tensor_track_allowed_irreps is None:
            self.tensor_track_allowed_irreps = o3.Irreps([
                (1, (l, p))
                for p in ((1, -1) if self.parity else (1,))
                for l in range(self.lmax + 1)
            ])
        else:
            self.tensor_track_allowed_irreps = o3.Irreps(self.tensor_track_allowed_irreps)

        # ----- Embeddings -----
        self.embeddings = nn.ModuleDict()
        self.embeddings["onehot_embedding"] = OneHotAtomEncoding(
            num_elements=num_types,
            species=species,
            universal_elems=self.universal_elems,
        )
        self.embeddings["radial_basis"] = RadialBasisEdgeEncoding(
            basis=BesselBasis(cutoff=self.cutoff, num_basis=self.num_basis),
            cutoff_fn=PolynomialCutoff(cutoff=self.cutoff, power=self.power),
        )
        self.embeddings["sphere_harmonics"] = SphericalHarmonicEdgeAttrs(
            edge_sh_irreps=self.edge_sh_irreps,
        )

        # ----- Two-body scalar embedding (ProductTypeEmbedding + scalar MLP) -----
        self.product_type_embedding = ProductTypeEmbedding(
            num_types=num_types,
            initial_embedding_dim=num_channels,
            radial_dim=self.num_basis,
            forward_weight_init=forward_weight_init,
        )
        self.scalar_embed_mlp = ScalarMLP(
            input_dim=num_channels,
            output_dim=num_scalar_features,
            hidden_layers_depth=two_body_mlp_hidden_layers_depth,
            hidden_layers_width=two_body_mlp_hidden_layers_width,
            nonlinearity=two_body_mlp_nonlinearity,
            bias=False,
            forward_weight_init=forward_weight_init,
        )

        # ----- Two-body tensor embedding -----
        self.tensor_embed = TwoBodySphericalHarmonicTensorEmbed(
            edge_sh_irreps=self.edge_sh_irreps,
            scalar_embed_dim=num_scalar_features,
            num_tensor_features=num_tensor_features,
            weight_individual_irreps=weight_individual_irreps,
            forward_weight_init=forward_weight_init,
        )

        # ----- Allegro interaction stack -----
        self.allegro_module = AllegroModule(
            num_layers=num_layers,
            num_scalar_features=num_scalar_features,
            num_tensor_features=num_tensor_features,
            scalar_input_dim=num_scalar_features,
            tensor_basis_irreps=self.edge_sh_irreps,
            tensor_track_allowed_irreps=self.tensor_track_allowed_irreps,
            avg_num_neighbors=avg_num_neighbors_init,
            weight_individual_irreps=weight_individual_irreps,
            latent_hidden_layers_depth=allegro_mlp_hidden_layers_depth,
            latent_hidden_layers_width=allegro_mlp_hidden_layers_width,
            latent_nonlinearity=allegro_mlp_nonlinearity,
            forward_weight_init=forward_weight_init,
            use_cue=self.use_cue,
        )

        # ----- Edge readout -----
        self.edge_readout = ScalarMLP(
            input_dim=self.allegro_module.scalar_out_dim,
            output_dim=1,
            hidden_layers_depth=readout_mlp_hidden_layers_depth,
            hidden_layers_width=readout_mlp_hidden_layers_width,
            nonlinearity=readout_mlp_nonlinearity,
            bias=False,
            forward_weight_init=forward_weight_init,
        )

        # ----- Edge -> atom reduction with sqrt(2) factor -----
        self.edgewise_reduce = EdgewiseReduce(avg_num_neighbors=avg_num_neighbors_init)

        # ----- Global energy normalization (legacy fallback) -----
        self.norm_data = nn.Parameter(torch.tensor(norm_data), requires_grad=False)
        self.norm_per_atom = nn.Parameter(torch.tensor(norm_per_atom), requires_grad=False)
        self.data_stddev = nn.Parameter(torch.tensor(data_stddev), requires_grad=False)
        self.data_mean = nn.Parameter(torch.tensor(data_mean), requires_grad=False)

        # ----- Per-type energy scale and shift -----
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

        self.atomwise_reduce = AtomwiseReduce(output_key="energy")

        self.compute_forces = bool(kwargs.get("compute_forces", False))
        self.compute_stress = bool(kwargs.get("compute_stress", False))
        self.compute_virial = bool(kwargs.get("compute_virial", False))

        outputs: List[str] = []
        if self.compute_forces:
            outputs.append("forces")
        if self.compute_stress:
            outputs.append("stress")
        if self.compute_virial:
            outputs.append("virial")

        if outputs:
            self.gradient_output: Optional[GradientOutput] = GradientOutput(
                model_outputs=outputs
            )
        else:
            self.gradient_output = None

    # ------------------------------------------------------------------
    # Internal stages
    # ------------------------------------------------------------------

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
        return replace_properties(
            data, edge_vectors=edge_vectors, displacement=displacement
        )

    def _edge_types(self, data: AtomsData) -> torch.Tensor:
        """Return per-edge ``(center_type, neighbor_type)`` indices into the
        ProductTypeEmbedding embeddings."""
        atomic_numbers = data.atomic_numbers
        edge_indices = data.edge_indices
        if self.universal_elems:
            atom_types = atomic_numbers - 1
        else:
            type_mapper = self.embeddings["onehot_embedding"].type_mapper
            if type_mapper is not None:
                atom_types = type_mapper.transform(atomic_numbers)
            else:
                atom_types = atomic_numbers - 1
        atom_types = atom_types.long()
        center = atom_types[edge_indices[:, 0]]
        neighbor = atom_types[edge_indices[:, 1]]
        return torch.stack([center, neighbor], dim=-1)

    def _forward_network(self, data: AtomsData) -> AtomsData:
        # 1. Run all base embeddings (one-hot, Bessel+cutoff, SH).
        for m in self.embeddings.values():
            data = m(data)

        # 2. Two-body scalar embedding (ProductTypeEmbedding + ScalarMLP).
        edge_types = self._edge_types(data)
        data = self.product_type_embedding(data, edge_types)
        scalar_embed = self.scalar_embed_mlp(data.edge_dist_embedding)
        data = replace_properties(data, edge_dist_embedding=scalar_embed)

        # 3. Two-body tensor embedding (learned-weighted SH).
        data, tensor_features = self.tensor_embed(data)

        # 4. Allegro interaction stack -> per-edge concatenated scalar track.
        data, final_scalars = self.allegro_module(data, scalar_embed, tensor_features)

        # 5. Per-edge readout -> per-edge scalar energy.
        edge_energy = self.edge_readout(final_scalars).squeeze(-1)

        # 6. Edge -> atom reduction with avg_num_neighbors and sqrt(2) norms.
        num_atoms = int(torch.sum(data.num_atoms))
        atomic_energy = self.edgewise_reduce(edge_energy, data.edge_indices, num_atoms)

        return replace_properties(data, atomic_energy=atomic_energy)

    def _apply_scale_shift(self, data: AtomsData) -> AtomsData:
        atomic_energy = data.atomic_energy
        per_type = self.per_type_scale_shift
        if per_type is not None:
            atomic_energy = per_type(atomic_energy, data.atomic_numbers.long())

        data = replace_properties(data, atomic_energy=atomic_energy)
        data = self.atomwise_reduce(data)

        if not self.use_per_type_scale_shift and bool(self.norm_data):
            energy = data.energy
            if energy is not None:
                normalizer = self.data_stddev
                energy = normalizer * energy
                if bool(self.norm_per_atom):
                    mean_shift = data.num_atoms.to(energy.dtype) * self.data_mean
                else:
                    mean_shift = self.data_mean
                energy = energy + mean_shift
                data = replace_properties(data, energy=energy)
        return data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, data: AtomsData) -> AtomsData:
        if self.compute_stress or self.compute_virial:
            data = self._apply_displacement(data)

        data = self._forward_network(data)
        data = self._apply_scale_shift(data)

        if self.gradient_output is not None:
            data = self.gradient_output(data)
        return data

    def datamodule(self, _datamodule):
        if hasattr(self.allegro_module, "datamodule"):
            self.allegro_module.datamodule(_datamodule)
        if hasattr(self.edgewise_reduce, "datamodule"):
            self.edgewise_reduce.datamodule(_datamodule)

    def get_optimization_info(self):
        return {
            "cuequivariance_available": _HAS_CUEQUIVARIANCE,
            "optimization_enabled": self.use_cue,
            "performance_boost": (
                "Allegro TPs always use e3nn (cuEq optimization not applicable)"
            ),
        }
