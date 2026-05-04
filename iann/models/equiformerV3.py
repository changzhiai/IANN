"""
EquiformerV3 model ported into the IANN framework.

The architecture follows the upstream reference implementation at
https://github.com/atomicarchitects/equiformer_v3/tree/main/experimental/models/equiformer_v3
with the following adaptations so that it plugs into IANN exactly like
``equiformerV2.py``:

* the ``forward`` consumes an ``AtomsData`` NamedTuple and returns one via
  ``replace_properties``;
* the precomputed graph (``edge_indices`` / ``edge_vectors``) is taken from
  ``AtomsData`` instead of computing it on-the-fly with ``GraphModelMixin``;
* forces, virial and stress are obtained through autograd via the
  ``GradientOutput`` module copied from ``equiformerV2.py``; the direct
  prediction heads are kept in the file but disabled by default;
* ``torch_geometric`` and ``torch_scatter`` are replaced with native PyTorch
  scatter ops so that the model has no extra runtime dependencies;
* ``Jd.pt`` is loaded from ``iann/data/Jd.pt`` (the file already shipped with
  IANN) instead of the V3 source directory.
"""
import os
import math
import copy
from functools import partial
from typing import List, Optional, Callable, Union

import torch
from torch import nn
from e3nn import o3
from e3nn.o3 import FromS2Grid, ToS2Grid

import iann
from iann.data import AtomsData, replace_properties


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Statistics of IS2RE 100K from the reference EquiformerV3
_AVG_NUM_NODES = 77.81317
_AVG_DEGREE = 23.395238876342773

_NORM_SCALE_NODES = math.sqrt(_AVG_NUM_NODES)
_NORM_SCALE_DEGREE = math.sqrt(_AVG_DEGREE)

# For gradient methods, do not back-propagate rotation if the y component of
# the edge unit vector is very close to this threshold.
_ROTATION_MASK_THRESHOLD = 0.999999


# ---------------------------------------------------------------------------
# Generic helpers (scatter / reduce / dropout / softmax)
# ---------------------------------------------------------------------------

def reduce_edge(inputs: torch.Tensor, edge_index: torch.Tensor, output_shape) -> torch.Tensor:
    outputs = torch.zeros(*output_shape, device=inputs.device, dtype=inputs.dtype)
    outputs.index_add_(0, edge_index, inputs)
    return outputs


def scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                    dim_size: Optional[int] = None,
                    exp_rescale: Optional[torch.Tensor] = None,
                    eps: float = 1e-16) -> torch.Tensor:
    """Numerically stable softmax over groups defined by ``index``.

    Mirrors ``torch_geometric.utils.softmax`` using native PyTorch ops so the
    model has no extra runtime dependency. ``exp_rescale`` lets us multiply
    the post-exp values (used by the envelope-rescaled attention).
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1

    index_expand = index
    for _ in range(src.dim() - index.dim()):
        index_expand = index_expand.unsqueeze(-1)
    index_expand = index_expand.expand_as(src)

    size = list(src.shape)
    size[dim] = dim_size

    max_value = torch.full(size, -1e10, dtype=src.dtype, device=src.device)
    max_value.scatter_reduce_(dim, index_expand, src.detach(), reduce='amax', include_self=False)

    out = src - max_value.gather(dim, index_expand)
    out = out.exp()

    if exp_rescale is not None:
        out = out * exp_rescale

    sum_value = torch.zeros(size, dtype=src.dtype, device=src.device)
    sum_value.scatter_add_(dim, index_expand, out)

    out = out / (sum_value.gather(dim, index_expand) + eps)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    """Lightweight ``scatter_mean`` reduction used by the (unused) stress heads."""
    if dim_size is None:
        dim_size = int(index.max()) + 1

    index_expand = index
    for _ in range(src.dim() - index.dim()):
        index_expand = index_expand.unsqueeze(-1)
    index_expand = index_expand.expand_as(src)

    size = list(src.shape)
    size[dim] = dim_size

    summed = torch.zeros(size, dtype=src.dtype, device=src.device)
    summed.scatter_add_(dim, index_expand, src)

    counts = torch.zeros(size, dtype=src.dtype, device=src.device)
    counts.scatter_add_(dim, index_expand, torch.ones_like(src))

    return summed / counts.clamp(min=1.0)


# ---------------------------------------------------------------------------
# Edge rotation matrices and Wigner-D
# ---------------------------------------------------------------------------

def init_edge_rot_mat(edge_distance_vec: torch.Tensor, use_rotation_mask: bool = False) -> torch.Tensor:
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0 ** 2, dim=1))

    if torch.min(edge_vec_0_distance) < 0.0001:
        print("Error edge_vec_0_distance: {}".format(torch.min(edge_vec_0_distance)))

    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

    if use_rotation_mask:
        yprod = norm_x @ norm_x.new_tensor([0.0, 1.0, 0.0])
        norm_x[yprod > _ROTATION_MASK_THRESHOLD] = norm_x.new_tensor([0.0, 1.0, 0.0])
        norm_x[yprod < -_ROTATION_MASK_THRESHOLD] = norm_x.new_tensor([0.0, -1.0, 0.0])

    edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
    edge_vec_2 = edge_vec_2 / (torch.sqrt(torch.sum(edge_vec_2 ** 2, dim=1)).view(-1, 1))

    edge_vec_2b = edge_vec_2.clone()
    edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.clone()
    edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    edge_vec_2c[:, 2] = edge_vec_2[:, 1]
    vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(-1, 1)
    vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(-1, 1)

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2)
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2)

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
    assert torch.max(vec_dot) < 0.99

    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / (torch.sqrt(torch.sum(norm_z ** 2, dim=1, keepdim=True)))
    norm_z = norm_z / (torch.sqrt(torch.sum(norm_z ** 2, dim=1)).view(-1, 1))
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / (torch.sqrt(torch.sum(norm_y ** 2, dim=1, keepdim=True)))

    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

    if use_rotation_mask:
        return edge_rot_mat
    return edge_rot_mat.detach()


# Borrowed from e3nn @ 0.4.0 (kept identical to upstream EquiformerV3)
_Jd = torch.load(os.path.join(iann.__path__[0], "data", "Jd.pt"))


def _z_rot_mat(angle: torch.Tensor, l: int) -> torch.Tensor:
    shape, device, dtype = angle.shape, angle.device, angle.dtype
    M = angle.new_zeros((*shape, 2 * l + 1, 2 * l + 1))
    inds = torch.arange(0, 2 * l + 1, 1, device=device)
    reversed_inds = torch.arange(2 * l, -1, -1, device=device)
    frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
    M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
    return M


def wigner_D(l: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    if not l < len(_Jd):
        raise NotImplementedError(
            f"wigner D maximum l implemented is {len(_Jd) - 1}, send us an email to ask for more"
        )
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[l].to(dtype=alpha.dtype, device=alpha.device)
    Xa = _z_rot_mat(alpha, l)
    Xb = _z_rot_mat(beta, l)
    Xc = _z_rot_mat(gamma, l)
    return Xa @ J @ Xb @ J @ Xc


# ---------------------------------------------------------------------------
# Radial functions and envelope
# ---------------------------------------------------------------------------

class PolynomialEnvelope(nn.Module):
    """Smooth polynomial cutoff envelope used on edge features."""
    def __init__(self, cutoff: float = 6.0, exponent: int = 5) -> None:
        super().__init__()
        assert exponent > 0
        self.cutoff = float(cutoff)
        self.exponent = exponent
        self.p: float = float(exponent)
        self.a: float = -(self.p + 1) * (self.p + 2) / 2
        self.b: float = self.p * (self.p + 2)
        self.c: float = -self.p * (self.p + 1) / 2

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        d_scaled = distance / self.cutoff
        env_val = (
            1
            + self.a * d_scaled ** self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        outputs = torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))
        outputs = outputs.view(-1, 1)
        return outputs

    def extra_repr(self):
        return 'cutoff={}, exponent={}'.format(self.cutoff, self.exponent)


class GaussianSmearing(nn.Module):
    def __init__(self, start: float = -5.0, stop: float = 5.0,
                 num_gaussians: int = 50, basis_width_scalar: float = 1.0) -> None:
        super().__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class RadialFunction(nn.Module):
    """Radial MLP optionally expanded over l/m components for SO(3)/SO(2)."""
    def __init__(self, channels_list, lmax=None, mmax=None,
                 use_rad_l_parametrization=True, use_expand=True):
        super().__init__()
        modules = []
        input_channels = channels_list[0]
        for i in range(len(channels_list)):
            if i == 0:
                continue
            modules.append(nn.Linear(input_channels, channels_list[i], bias=True))
            input_channels = channels_list[i]
            if i == len(channels_list) - 1:
                break
            modules.append(nn.LayerNorm(channels_list[i]))
            modules.append(nn.SiLU())
        self.net = nn.Sequential(*modules)

        self.lmax = lmax
        self.mmax = mmax
        self.use_rad_l_parametrization = use_rad_l_parametrization
        self.use_expand = use_expand

        if self.use_expand:
            if not self.use_rad_l_parametrization:
                expand_index = []
                offset = 0
                for m in range(self.mmax + 1):
                    index = torch.arange((self.lmax + 1 - m))
                    index = index + offset
                    expand_index.append(index)
                    if m > 0:
                        expand_index.append(index)
                    offset = offset + len(index)
                expand_index = torch.cat(expand_index, dim=0).long()
                self.register_buffer('expand_index', expand_index)
                self.num_m_components = offset
                assert channels_list[-1] % self.num_m_components == 0
            else:
                assert self.lmax == self.mmax
                expand_index = torch.zeros([((self.lmax + 1) ** 2)]).long()
                start_idx = 0
                for l in range(self.lmax + 1):
                    length = 2 * l + 1
                    expand_index[start_idx: (start_idx + length)] = l
                    start_idx = start_idx + length
                self.register_buffer('expand_index', expand_index)
                assert channels_list[-1] % (self.lmax + 1) == 0

    def forward(self, inputs):
        outputs = self.net(inputs)
        if self.use_expand:
            if not self.use_rad_l_parametrization:
                outputs = outputs.view(outputs.shape[0], self.num_m_components, -1)
            else:
                outputs = outputs.view(outputs.shape[0], (self.lmax + 1), -1)
            outputs = torch.index_select(outputs, dim=1, index=self.expand_index)
        return outputs


# ---------------------------------------------------------------------------
# Coefficient mapping / SO(3) primitives
# ---------------------------------------------------------------------------

class CoefficientMappingModule(nn.Module):
    """Helper for reshaping irreps between l-major and m-major layouts."""
    def __init__(self, lmax, mmax, use_rotate_inv_rescale=False):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.use_rotate_inv_rescale = use_rotate_inv_rescale

        l_harmonic = []
        m_harmonic = []
        m_complex = []

        for l in range(0, self.lmax + 1):
            mmax_l = min(self.mmax, l)
            m = torch.arange(-mmax_l, mmax_l + 1).long()
            m_complex.append(m)
            m_harmonic.append(torch.abs(m).long())
            l_harmonic.append(torch.fill(m, l))
        m_complex = torch.cat(m_complex, dim=0)
        m_harmonic = torch.cat(m_harmonic, dim=0)
        l_harmonic = torch.cat(l_harmonic, dim=0)

        num_m_coefficients = len(l_harmonic)
        to_m = torch.zeros([num_m_coefficients, num_m_coefficients])

        offset = 0
        for m in range(self.mmax + 1):
            idx_r, idx_i = self.complex_idx(m, -1, m_complex, l_harmonic)
            for idx_out, idx_in in enumerate(idx_r):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_r)
            for idx_out, idx_in in enumerate(idx_i):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_i)

        to_m = to_m.detach()

        self.register_buffer('l_harmonic', l_harmonic)
        self.register_buffer('m_harmonic', m_harmonic)
        self.register_buffer('m_complex', m_complex)
        self.register_buffer('to_m', to_m)

        self.pre_compute_coefficient_idx()
        if self.use_rotate_inv_rescale:
            self.pre_compute_rotate_inv_rescale()

    def complex_idx(self, m, lmax, m_complex, l_harmonic):
        if lmax == -1:
            lmax = self.lmax
        indices = torch.arange(len(l_harmonic))
        mask_r = torch.bitwise_and(l_harmonic.le(lmax), m_complex.eq(m))
        mask_idx_r = torch.masked_select(indices, mask_r)
        mask_idx_i = torch.tensor([]).long()
        if m != 0:
            mask_i = torch.bitwise_and(l_harmonic.le(lmax), m_complex.eq(-m))
            mask_idx_i = torch.masked_select(indices, mask_i)
        return mask_idx_r, mask_idx_i

    def pre_compute_coefficient_idx(self):
        for l in range(self.lmax + 1):
            for m in range(self.lmax + 1):
                mask = torch.bitwise_and(self.l_harmonic.le(l), self.m_harmonic.le(m))
                indices = torch.arange(len(mask))
                mask_indices = torch.masked_select(indices, mask)
                self.register_buffer('coefficient_idx_l{}_m{}'.format(l, m), mask_indices)

    def prepare_coefficient_idx(self):
        coefficient_idx_list = []
        for l in range(self.lmax + 1):
            l_list = []
            for m in range(self.lmax + 1):
                l_list.append(getattr(self, 'coefficient_idx_l{}_m{}'.format(l, m), None))
            coefficient_idx_list.append(l_list)
        return coefficient_idx_list

    def coefficient_idx(self, lmax, mmax):
        if lmax > self.lmax or mmax > self.lmax:
            mask = torch.bitwise_and(self.l_harmonic.le(lmax), self.m_harmonic.le(mmax))
            indices = torch.arange(len(mask), device=mask.device)
            return torch.masked_select(indices, mask)
        temp = self.prepare_coefficient_idx()
        return temp[lmax][mmax]

    def pre_compute_rotate_inv_rescale(self):
        for l in range(self.lmax + 1):
            for m in range(self.lmax + 1):
                mask_indices = self.coefficient_idx(l, m)
                rotate_inv_rescale = torch.ones((1, int((l + 1) ** 2), int((l + 1) ** 2)))
                for l_sub in range(l + 1):
                    if l_sub <= m:
                        continue
                    start_idx = l_sub ** 2
                    length = 2 * l_sub + 1
                    rescale_factor = math.sqrt(length / (2 * m + 1))
                    rotate_inv_rescale[:, start_idx:(start_idx + length), start_idx:(start_idx + length)] = rescale_factor
                rotate_inv_rescale = rotate_inv_rescale[:, :, mask_indices]
                self.register_buffer('rotate_inv_rescale_l{}_m{}'.format(l, m), rotate_inv_rescale)

    def prepare_rotate_inv_rescale(self):
        rotate_inv_rescale_list = []
        for l in range(self.lmax + 1):
            l_list = []
            for m in range(self.lmax + 1):
                l_list.append(getattr(self, 'rotate_inv_rescale_l{}_m{}'.format(l, m), None))
            rotate_inv_rescale_list.append(l_list)
        return rotate_inv_rescale_list

    def get_rotate_inv_rescale(self, lmax, mmax):
        temp = self.prepare_rotate_inv_rescale()
        return temp[lmax][mmax]

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, mmax={self.mmax})"


class SO3Rotation(nn.Module):
    """Wigner-D rotations baked together with the m-primary layout swap."""
    def __init__(self, lmax, mmax, use_rotation_mask=False):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.use_rotation_mask = use_rotation_mask

        mapping = CoefficientMappingModule(
            lmax=self.lmax, mmax=self.lmax, use_rotate_inv_rescale=True
        )
        wigner_index_mask = mapping.coefficient_idx(self.lmax, self.mmax)
        wigner_inv_rescale = mapping.get_rotate_inv_rescale(self.lmax, self.mmax)

        mapping = CoefficientMappingModule(
            lmax=self.lmax, mmax=self.mmax, use_rotate_inv_rescale=False
        )
        to_m = mapping.to_m
        wigner_inv_rescale = torch.einsum('nia, ba -> nib', wigner_inv_rescale, to_m)
        wigner_index_to_m_array = torch.zeros(
            to_m.shape[0], ((self.lmax + 1) ** 2)
        )
        wigner_index_to_m_array[:, wigner_index_mask] = to_m

        self.register_buffer('wigner_index_to_m_array', wigner_index_to_m_array)
        self.register_buffer('wigner_inv_rescale', wigner_inv_rescale)

        self.wigner: Optional[torch.Tensor] = None
        self.wigner_inv: Optional[torch.Tensor] = None

    def set_wigner(self, rot_mat3x3: torch.Tensor) -> None:
        wigner = self._rotation_to_wigner_matrix(rot_mat3x3, 0, self.lmax)
        wigner = torch.einsum('mi, nij -> nmj', self.wigner_index_to_m_array, wigner)
        if torch.is_autocast_enabled():
            wigner = wigner.to(torch.float16)
        wigner_inv = torch.transpose(wigner, 1, 2).contiguous()
        wigner_inv = wigner_inv * self.wigner_inv_rescale
        if torch.is_autocast_enabled():
            wigner_inv = wigner_inv.to(torch.float16)
        self.wigner = wigner
        self.wigner_inv = wigner_inv

    def rotate(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.bmm(self.wigner, inputs)

    def rotate_inv(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.bmm(self.wigner_inv, inputs)

    def _rotation_to_wigner_matrix(self, edge_rot_mat: torch.Tensor,
                                   start_lmax: int, end_lmax: int) -> torch.Tensor:
        x = edge_rot_mat[:, :, 1]
        alpha, beta = o3.xyz_to_angles(x)
        R = o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2)
        R = torch.bmm(R, edge_rot_mat)
        gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])

        backprop_mask = None
        alpha_detach = beta_detach = gamma_detach = None
        if self.use_rotation_mask:
            yprod = (x @ x.new_tensor([0, 1, 0])).detach()
            backprop_mask = (yprod > -_ROTATION_MASK_THRESHOLD) & (yprod < _ROTATION_MASK_THRESHOLD)
            alpha_detach = alpha[(~backprop_mask)].clone().detach()
            gamma_detach = gamma[(~backprop_mask)].clone().detach()
            beta_detach = beta.clone().detach()
            beta_detach[yprod > _ROTATION_MASK_THRESHOLD] = 0.0
            beta_detach[yprod < -_ROTATION_MASK_THRESHOLD] = math.pi
            beta_detach = beta_detach[(~backprop_mask)]

        size = int((end_lmax + 1) ** 2) - int((start_lmax) ** 2)
        wigner = torch.zeros(len(alpha), size, size, device=edge_rot_mat.device)
        start = 0
        end = 0
        for lmax in range(start_lmax, end_lmax + 1):
            if self.use_rotation_mask:
                block = wigner_D(lmax, alpha[backprop_mask], beta[backprop_mask], gamma[backprop_mask])
                block_detach = wigner_D(lmax, alpha_detach, beta_detach, gamma_detach)
                end = start + block.size()[1]
                wigner[backprop_mask, start:end, start:end] = block
                wigner[(~backprop_mask), start:end, start:end] = block_detach
            else:
                block = wigner_D(lmax, alpha, beta, gamma)
                end = start + block.size()[1]
                wigner[:, start:end, start:end] = block
            start = end
        if self.use_rotation_mask:
            return wigner
        return wigner.detach()

    def extra_repr(self):
        return 'lmax={}, mmax={}'.format(self.lmax, self.mmax)


class SO3Grid(nn.Module):
    """Spherical harmonic <-> S2 grid conversion (component-normalized)."""
    def __init__(self, lmax, mmax, normalization='component',
                 resolution_list=None, use_m_primary=False):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.use_m_primary = use_m_primary
        self.lat_resolution = 2 * (self.lmax + 1)
        if lmax == mmax:
            self.long_resolution = 2 * (self.mmax + 1) + 1
        else:
            self.long_resolution = 2 * (self.mmax) + 1
        if resolution_list is not None:
            assert isinstance(resolution_list, list)
            resolution_list = copy.deepcopy(resolution_list)
            self.lat_resolution = resolution_list[0]
            self.long_resolution = resolution_list[1]

        mapping = CoefficientMappingModule(
            lmax=self.lmax, mmax=self.lmax, use_rotate_inv_rescale=False
        )

        to_grid = ToS2Grid(
            self.lmax,
            (self.lat_resolution, self.long_resolution),
            normalization=normalization,
            device='cpu',
        )
        to_grid_mat = torch.einsum("mbi, am -> bai", to_grid.shb, to_grid.sha).detach()
        if lmax != mmax:
            for l in range(lmax + 1):
                if l <= mmax:
                    continue
                start_idx = l ** 2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                to_grid_mat[:, :, start_idx:(start_idx + length)] = to_grid_mat[:, :, start_idx:(start_idx + length)] * rescale_factor
        to_grid_mat = to_grid_mat[:, :, mapping.coefficient_idx(self.lmax, self.mmax)]

        from_grid = FromS2Grid(
            (self.lat_resolution, self.long_resolution),
            self.lmax,
            normalization=normalization,
            device='cpu',
        )
        from_grid_mat = torch.einsum("am, mbi -> bai", from_grid.sha, from_grid.shb).detach()
        if lmax != mmax:
            for l in range(lmax + 1):
                if l <= mmax:
                    continue
                start_idx = l ** 2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                from_grid_mat[:, :, start_idx:(start_idx + length)] = from_grid_mat[:, :, start_idx:(start_idx + length)] * rescale_factor
        from_grid_mat = from_grid_mat[:, :, mapping.coefficient_idx(self.lmax, self.mmax)]

        to_grid_mat = to_grid_mat.flatten(0, 1)
        from_grid_mat = from_grid_mat.flatten(0, 1)
        from_grid_mat = from_grid_mat.permute(1, 0)

        if self.use_m_primary:
            temp = CoefficientMappingModule(self.lmax, self.mmax, False)
            to_grid_mat = torch.einsum('ai, ji -> aj', to_grid_mat, temp.to_m)
            from_grid_mat = torch.einsum('ia, ji -> ja', from_grid_mat, temp.to_m)

        self.register_buffer('to_grid_mat', to_grid_mat)
        self.register_buffer('from_grid_mat', from_grid_mat)

    def get_to_grid_mat(self):
        return self.to_grid_mat

    def get_from_grid_mat(self):
        return self.from_grid_mat

    def to_grid(self, embedding: torch.Tensor) -> torch.Tensor:
        return torch.einsum('aj, njc -> nac', self.to_grid_mat, embedding)

    def from_grid(self, grid: torch.Tensor) -> torch.Tensor:
        return torch.einsum('ja, nac -> njc', self.from_grid_mat, grid)

    def extra_repr(self):
        return 'lmax={}, mmax={}, lat_resolution={}, long_resolution={}, use_m_primary={}'.format(
            self.lmax, self.mmax, self.lat_resolution, self.long_resolution, self.use_m_primary
        )


class SO3Linear(nn.Module):
    """l-wise linear: a separate dense map per l, broadcast over m."""
    def __init__(self, in_features, out_features, lmax, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax

        self.weight = nn.Parameter(torch.randn((self.lmax + 1), out_features, in_features))
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        self.bias = nn.Parameter(torch.zeros(1, 1, out_features)) if bias else None

        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for l in range(lmax + 1):
            start_idx = l ** 2
            length = 2 * l + 1
            expand_index[start_idx:(start_idx + length)] = l
        self.register_buffer('expand_index', expand_index)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = torch.index_select(self.weight, dim=0, index=self.expand_index)
        outputs = torch.einsum('bmi, moi -> bmo', inputs, weight)
        if self.bias is not None:
            outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + self.bias
        return outputs

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, lmax={self.lmax}, bias={(self.bias is not None)})"


# ---------------------------------------------------------------------------
# SO(2) operations
# ---------------------------------------------------------------------------

class SO2MLinear(nn.Module):
    """SO(2) linear over the +-m features for a fixed m."""
    def __init__(self, m, num_in_channels, num_out_channels, lmax, mmax):
        super().__init__()
        self.m = m
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.lmax = lmax
        self.mmax = mmax

        num_m_components = self.lmax - self.m + 1
        assert num_m_components > 0

        self.in_features = num_m_components * self.num_in_channels
        self.out_features = num_m_components * self.num_out_channels

        self.fc = nn.Linear(self.in_features, (2 * self.out_features), bias=False)
        self.fc.weight.data.mul_(1 / math.sqrt(2))

    def forward(self, x_m, concat_outputs=True):
        x_m = self.fc(x_m)
        x_r = x_m.narrow(2, 0, self.out_features)
        x_i = x_m.narrow(2, self.out_features, self.out_features)
        x_m_r = x_r.narrow(1, 0, 1) - x_i.narrow(1, 1, 1)
        x_m_i = x_r.narrow(1, 1, 1) + x_i.narrow(1, 0, 1)
        x_out = (x_m_r, x_m_i)
        if concat_outputs:
            x_out = torch.cat(x_out, dim=1)
        return x_out


class SO2Linear(nn.Module):
    """SO(2) linear over all m components, with an optional extra m=0 head."""
    def __init__(self, num_in_channels, num_out_channels, lmax, mmax,
                 extra_m0_out_channels=None):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.lmax = lmax
        self.mmax = mmax
        self.extra_m0_out_channels = extra_m0_out_channels

        num_in_channels_m0 = (self.lmax + 1) * self.num_in_channels
        num_out_channels_m0 = (self.lmax + 1) * self.num_out_channels
        if self.extra_m0_out_channels is not None:
            self.num_channels_m0_list = [self.extra_m0_out_channels, num_out_channels_m0]
            num_out_channels_m0 = num_out_channels_m0 + self.extra_m0_out_channels
        self.fc_m0 = nn.Linear(num_in_channels_m0, num_out_channels_m0)

        self.so2_m_linear = nn.ModuleList()
        for m in range(1, self.mmax + 1):
            self.so2_m_linear.append(
                SO2MLinear(m, self.num_in_channels, self.num_out_channels, self.lmax, self.mmax)
            )

    def forward(self, x):
        num_edges = x.shape[0]
        outputs = []

        x_m0 = x.narrow(1, 0, (self.lmax + 1))
        x_m0 = x_m0.reshape(num_edges, -1)
        x_m0 = self.fc_m0(x_m0)

        x_m0_extra = None
        if self.extra_m0_out_channels is not None:
            x_m0_extra, x_m0 = torch.split(x_m0, self.num_channels_m0_list, dim=1)

        x_m0 = x_m0.view(num_edges, -1, self.num_out_channels)
        outputs.append(x_m0)

        offset = self.lmax + 1
        for m in range(1, self.mmax + 1):
            x_m = x.narrow(1, offset, 2 * (self.lmax + 1 - m))
            offset = offset + 2 * (self.lmax + 1 - m)
            x_m = x_m.reshape(num_edges, 2, -1)
            x_m = self.so2_m_linear[m - 1](x_m, concat_outputs=False)
            x_m_pos, x_m_neg = x_m[0], x_m[1]
            x_m_pos = x_m_pos.view(num_edges, -1, self.num_out_channels)
            x_m_neg = x_m_neg.view(num_edges, -1, self.num_out_channels)
            outputs.append(x_m_pos)
            outputs.append(x_m_neg)

        outputs = torch.cat(outputs, dim=1)

        if self.extra_m0_out_channels is not None:
            return outputs, x_m0_extra
        return outputs


# ---------------------------------------------------------------------------
# Normalization layers
# ---------------------------------------------------------------------------

_NORM_TYPE_LIST = [
    'equivariant_layer_norm',
    'sep_layer_norm',
    'merge_layer_norm',
    'merge_layer_norm_attn_rms_norm',
    'merge_rms_norm',
]


class EquivariantLayerNorm(nn.Module):
    def __init__(self, lmax, num_channels, eps=1e-5, affine=True, normalization='component'):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.affine_weight = nn.Parameter(torch.ones((self.lmax + 1), self.num_channels))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_channels))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps})"

    def forward(self, inputs):
        outputs = []
        for l in range(self.lmax + 1):
            start_idx = l ** 2
            length = 2 * l + 1
            feature = inputs.narrow(1, start_idx, length)

            if l == 0:
                feature_mean = torch.mean(feature, dim=2, keepdim=True)
                feature = feature - feature_mean

            if self.normalization == 'norm':
                feature_norm = feature.pow(2).sum(dim=1, keepdim=True)
            elif self.normalization == 'component':
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)

            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)
            feature_norm = (feature_norm + self.eps).pow(-0.5)

            if self.affine:
                weight = self.affine_weight.narrow(0, l, 1)
                weight = weight.view(1, 1, -1)
                feature_norm = feature_norm * weight

            feature = feature * feature_norm

            if self.affine and l == 0:
                bias = self.affine_bias
                bias = bias.view(1, 1, -1)
                feature = feature + bias

            outputs.append(feature)

        return torch.cat(outputs, dim=1)


class EquivariantSeparableLayerNorm(nn.Module):
    def __init__(self, lmax, num_channels, eps=1e-5, affine=True,
                 normalization='component', std_balance_degrees=True):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.std_balance_degrees = std_balance_degrees

        self.norm_l0 = nn.LayerNorm(self.num_channels, eps=self.eps, elementwise_affine=self.affine)

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.lmax, self.num_channels))
            expand_index = torch.zeros([((self.lmax + 1) ** 2 - 1)]).long()
            for l in range(1, self.lmax + 1):
                start_idx = l ** 2 - 1
                length = 2 * l + 1
                expand_index[start_idx:(start_idx + length)] = (l - 1)
            self.register_buffer('expand_index', expand_index)
        else:
            self.register_parameter('affine_weight', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2 - 1, 1)
            for l in range(1, self.lmax + 1):
                start_idx = l ** 2 - 1
                length = 2 * l + 1
                balance_degree_weight[start_idx:(start_idx + length), :] = (1.0 / length)
            balance_degree_weight = balance_degree_weight / self.lmax
            balance_degree_weight = balance_degree_weight.permute((1, 0))
            self.register_buffer('balance_degree_weight', balance_degree_weight)
        else:
            self.balance_degree_weight = None

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, std_balance_degrees={self.std_balance_degrees})"

    def forward(self, inputs):
        outputs = []
        scalars = inputs.narrow(1, 0, 1)
        scalars = self.norm_l0(scalars)
        outputs.append(scalars)

        if self.lmax > 0:
            num_m_components = (self.lmax + 1) ** 2
            feature = inputs.narrow(1, 1, num_m_components - 1)

            feature_norm = feature.pow(2)
            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)

            if self.normalization == 'norm':
                feature_norm = feature_norm.sum(dim=1, keepdim=True)
            elif self.normalization == 'component':
                if self.std_balance_degrees:
                    feature_norm = torch.einsum('ai, nic -> nac', self.balance_degree_weight, feature_norm)
                else:
                    feature_norm = feature_norm.mean(dim=1, keepdim=True)

            feature_norm = (feature_norm + self.eps).pow(-0.5)

            if self.affine:
                weight = self.affine_weight.view(1, self.lmax, self.num_channels)
                weight = torch.index_select(weight, dim=1, index=self.expand_index)
                feature_norm = feature_norm * weight
            feature = feature * feature_norm

            outputs.append(feature)

        return torch.cat(outputs, dim=1)


class EquivariantMergeLayerNorm(nn.Module):
    def __init__(self, lmax, num_channels, eps=1e-5, affine=True,
                 normalization='component', std_balance_degrees=True, centering=True):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.std_balance_degrees = std_balance_degrees
        self.centering = centering

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones((self.lmax + 1), self.num_channels))
            expand_index = torch.zeros([((self.lmax + 1) ** 2)]).long()
            for l in range(self.lmax + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                expand_index[start_idx:(start_idx + length)] = l
            self.register_buffer('expand_index', expand_index)

            if self.centering:
                self.affine_bias = nn.Parameter(torch.zeros(self.num_channels))
            else:
                self.register_parameter('affine_bias', None)
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2, 1)
            for l in range(self.lmax + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                balance_degree_weight[start_idx:(start_idx + length), :] = (1.0 / length)
            balance_degree_weight = balance_degree_weight / (self.lmax + 1)
            balance_degree_weight = balance_degree_weight.permute((1, 0))
            self.register_buffer('balance_degree_weight', balance_degree_weight)
        else:
            self.balance_degree_weight = None

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, std_balance_degrees={self.std_balance_degrees}, centering={self.centering})"

    def forward(self, inputs):
        if self.centering:
            scalars = inputs.narrow(1, 0, 1)
            scalars_mean = scalars.mean(dim=2, keepdim=True)
            scalars = scalars - scalars_mean
            inputs = torch.cat((scalars, inputs.narrow(1, 1, inputs.shape[1] - 1)), dim=1)

        feature_norm = inputs.pow(2)
        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)
        if self.normalization == 'norm':
            feature_norm = feature_norm.sum(dim=1, keepdim=True)
        elif self.normalization == 'component':
            if self.std_balance_degrees:
                feature_norm = torch.einsum('ai, nic -> nac', self.balance_degree_weight, feature_norm)
            else:
                feature_norm = feature_norm.mean(dim=1, keepdim=True)
        feature_norm = (feature_norm + self.eps).pow(-0.5)
        if self.affine:
            weight = self.affine_weight.view(1, (self.lmax + 1), self.num_channels)
            weight = torch.index_select(weight, dim=1, index=self.expand_index)
            feature_norm = feature_norm * weight
        outputs = inputs * feature_norm

        if self.affine and self.centering:
            outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + self.affine_bias.view(1, 1, self.num_channels)

        return outputs


class RMSNorm(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.num_channels))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def __repr__(self):
        return f"{self.__class__.__name__}(num_channels={self.num_channels}, eps={self.eps})"


def get_normalization_layer(norm_type, lmax, num_channels, eps=1e-5,
                            affine=True, normalization='component'):
    assert norm_type in _NORM_TYPE_LIST
    if norm_type == 'equivariant_layer_norm':
        norm_class = EquivariantLayerNorm
    elif norm_type == 'sep_layer_norm':
        norm_class = EquivariantSeparableLayerNorm
    elif norm_type in ['merge_layer_norm', 'merge_layer_norm_attn_rms_norm']:
        norm_class = EquivariantMergeLayerNorm
    elif norm_type == 'merge_rms_norm':
        norm_class = partial(EquivariantMergeLayerNorm, centering=False)
    else:
        raise ValueError
    return norm_class(lmax, num_channels, eps, affine, normalization)


# ---------------------------------------------------------------------------
# Activations (gate / S2 / SwiGLU / merge variants)
# ---------------------------------------------------------------------------

def check_activation_name(act_name):
    assert act_name in [
        'gate',
        's2',
        'sep_s2',
        's2_swiglu',
        's2_swiglu_mem',
        'sep_s2_swiglu',
        'sep-merge_s2_swiglu',
        'sep_s2_swiglu_mem',
        'sep-merge_s2_swiglu_mem',
        'sep_s2_square',
        'sep-merge_gates2_swiglu',
        'sep-merge_gates2_swiglu_mem',
    ]


def has_scalars(act_name):
    if act_name not in ['s2', 's2_swiglu', 's2_swiglu_mem']:
        return True
    return False


def prepare_activation_forward_param(act_name, inputs, scalars):
    output_dict = {'inputs': inputs}
    if has_scalars(act_name):
        output_dict['scalars'] = scalars
    return output_dict


class SmoothLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.alpha = negative_slope

    def forward(self, x):
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2

    def extra_repr(self):
        return 'negative_slope={}'.format(self.alpha)


class GateActivation(nn.Module):
    def __init__(self, lmax, mmax, use_m_primary=False):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.use_m_primary = use_m_primary

        num_components = 0
        for l in range(1, self.lmax + 1):
            num_m_components = min((2 * l + 1), (2 * self.mmax + 1))
            num_components = num_components + num_m_components
        if not self.use_m_primary:
            expand_index = torch.zeros([num_components]).long()
            start_idx = 0
            for l in range(1, self.lmax + 1):
                length = min((2 * l + 1), (2 * self.mmax + 1))
                expand_index[start_idx:(start_idx + length)] = (l - 1)
                start_idx = start_idx + length
        else:
            expand_index = []
            for m in range(self.mmax + 1):
                if m == 0:
                    l_index = torch.arange(self.lmax)
                else:
                    l_index = torch.arange((m - 1), self.lmax)
                expand_index.append(l_index)
                if m > 0:
                    expand_index.append(l_index)
            expand_index = torch.cat(expand_index, dim=0).long()
        self.register_buffer('expand_index', expand_index)

        self.scalar_act = nn.SiLU()
        self.gate_act = nn.Sigmoid()

    def forward(self, inputs, scalars):
        gate_scalars = self.gate_act(scalars)
        gate_scalars = gate_scalars.reshape(gate_scalars.shape[0], self.lmax, -1)
        gate_scalars = torch.index_select(gate_scalars, dim=1, index=self.expand_index)

        input_scalars = inputs.narrow(1, 0, 1)
        input_scalars = self.scalar_act(input_scalars)
        input_vectors = inputs.narrow(1, 1, inputs.shape[1] - 1)
        input_vectors = input_vectors * gate_scalars

        return torch.cat((input_scalars, input_vectors), dim=1)

    def extra_repr(self):
        return 'lmax={}, mmax={}, use_m_primary={}'.format(self.lmax, self.mmax, self.use_m_primary)


class S2Activation(nn.Module):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.so3_grid = SO3Grid(self.lmax, self.mmax, resolution_list=grid_resolution_list, use_m_primary=use_m_primary)
        self.act = nn.SiLU()

    def forward(self, inputs):
        x_grid = self.so3_grid.to_grid(inputs)
        x_grid = self.act(x_grid)
        return self.so3_grid.from_grid(x_grid)


class SeparableS2Activation(S2Activation):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary)

    def forward(self, inputs, scalars):
        output_scalars = self.act(scalars)
        output_scalars = output_scalars.reshape(output_scalars.shape[0], 1, output_scalars.shape[1])
        output_vectors = super().forward(inputs)
        return torch.cat(
            (output_scalars, output_vectors.narrow(1, 1, output_vectors.shape[1] - 1)),
            dim=1,
        )


def swiglu_torch(gate, up_states):
    gate = nn.functional.silu(gate)
    return gate * up_states


class SwiGLU(nn.Module):
    def __init__(self, backend='torch'):
        super().__init__()
        assert backend in ['torch']
        self.backend = backend
        self.func = swiglu_torch

    def forward(self, inputs):
        x_1, x_2 = torch.chunk(inputs, chunks=2, dim=-1)
        return self.func(x_1, x_2)

    def extra_repr(self):
        return 'backend={}'.format(self.backend)


class LinearSwiGLU(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, backend='torch'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, 2 * out_channels, bias=bias)
        self.act = SwiGLU(backend)

    def forward(self, inputs):
        return self.act(self.linear(inputs))


class S2Activation_SwiGLU(S2Activation):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary)
        del self.act
        self.act = SwiGLU(backend)


class S2Activation_SwiGLU_MemoryEfficient(S2Activation_SwiGLU):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary, backend)

    def kernel(self, inputs):
        x_grid = self.so3_grid.to_grid(inputs)
        x_grid = self.act(x_grid)
        return self.so3_grid.from_grid(inputs)

    def forward(self, inputs):
        return torch.utils.checkpoint.checkpoint(self.kernel, inputs, use_reentrant=False)


class SeparableS2Activation_SwiGLU(S2Activation_SwiGLU):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary, backend)

    def forward(self, inputs, scalars):
        output_scalars = self.act(scalars)
        output_scalars = output_scalars.reshape(output_scalars.shape[0], 1, output_scalars.shape[1])
        output_vectors = super().forward(inputs)
        return torch.cat(
            (output_scalars, output_vectors.narrow(1, 1, output_vectors.shape[1] - 1)),
            dim=1,
        )


class SeparableS2Activation_SwiGLU_Merge(S2Activation_SwiGLU):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary, backend)

    def forward(self, inputs, scalars):
        output_scalars = self.act(scalars)
        output_scalars = output_scalars.reshape(output_scalars.shape[0], 1, output_scalars.shape[1])
        output_vectors = super().forward(inputs)
        outputs = output_vectors
        outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + output_scalars
        return outputs


class SeparableS2Activation_SwiGLU_MemoryEfficient(S2Activation_SwiGLU_MemoryEfficient):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary, backend)

    def forward(self, inputs, scalars):
        output_scalars = self.act(scalars)
        output_scalars = output_scalars.reshape(output_scalars.shape[0], 1, output_scalars.shape[1])
        output_vectors = super().forward(inputs)
        return torch.cat(
            (output_scalars, output_vectors.narrow(1, 1, output_vectors.shape[1] - 1)),
            dim=1,
        )


class SeparableS2Activation_SwiGLU_Merge_MemoryEfficient(S2Activation_SwiGLU_MemoryEfficient):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary, backend)

    def kernel(self, inputs, scalars):
        x_grid = self.so3_grid.to_grid(inputs)
        x_grid = self.act(x_grid)
        output_vectors = self.so3_grid.from_grid(x_grid)
        output_scalars = self.act(scalars)
        output_scalars = output_scalars.reshape(output_scalars.shape[0], 1, output_scalars.shape[-1])
        outputs = output_vectors
        outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + output_scalars
        return outputs

    def forward(self, inputs, scalars):
        return torch.utils.checkpoint.checkpoint(self.kernel, inputs, scalars, use_reentrant=False)


class Square(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        x_1, x_2 = torch.chunk(inputs, chunks=2, dim=-1)
        return x_1 * x_2


class LinearSquare(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, 2 * out_channels, bias=bias)
        self.act = Square()

    def forward(self, inputs):
        return self.act(self.linear(inputs))


class SeparableS2Activation_Square(nn.Module):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.so3_grid = SO3Grid(self.lmax, self.mmax, resolution_list=grid_resolution_list, use_m_primary=use_m_primary)
        self.act = Square()

    def forward(self, inputs, scalars):
        x_grid = self.so3_grid.to_grid(inputs)
        x_grid = self.act(x_grid)
        output_vectors = self.so3_grid.from_grid(x_grid)
        output_scalars = self.act(scalars)
        output_scalars = output_scalars.reshape(output_scalars.shape[0], 1, output_scalars.shape[-1])
        return torch.cat(
            (output_scalars, output_vectors.narrow(1, 1, output_vectors.shape[1] - 1)),
            dim=1,
        )


class SeparableGateS2Activation_SwiGLU_Merge(GateActivation):
    """The default V3 attention/FFN activation: gated SwiGLU on grid signals,
    merged with a SwiGLU scalar path."""
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, use_m_primary)
        self.so3_grid = SO3Grid(self.lmax, self.mmax, resolution_list=grid_resolution_list, use_m_primary=use_m_primary)
        del self.scalar_act
        self.scalar_act = SwiGLU(backend)
        del self.expand_index
        self.grid_drop = nn.Identity()

    def forward(self, inputs, scalars):
        scalars = scalars.view(scalars.shape[0], 1, scalars.shape[-1])
        output_scalars = scalars.narrow(2, 0, inputs.shape[2])
        gate_scalars = scalars.narrow(2, output_scalars.shape[2], (scalars.shape[2] - output_scalars.shape[2]))
        output_scalars = self.scalar_act(output_scalars)
        gate_scalars = self.gate_act(gate_scalars)
        x_grid = self.so3_grid.to_grid(inputs)
        x_grid_1, x_grid_2 = torch.chunk(x_grid, chunks=2, dim=-1)
        x_grid = x_grid_1 * x_grid_2
        x_grid = self.grid_drop(x_grid)
        output_vectors = self.so3_grid.from_grid(x_grid)
        output_vectors = output_vectors * gate_scalars
        outputs = output_vectors
        outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + output_scalars
        return outputs


class SeparableGateS2Activation_SwiGLU_Merge_MemoryEfficient(SeparableGateS2Activation_SwiGLU_Merge):
    def __init__(self, lmax, mmax, grid_resolution_list=None, use_m_primary=False, backend='torch'):
        super().__init__(lmax, mmax, grid_resolution_list, use_m_primary, backend)

    def kernel(self, inputs, scalars):
        scalars = scalars.view(scalars.shape[0], 1, scalars.shape[-1])
        output_scalars = scalars.narrow(2, 0, inputs.shape[2])
        gate_scalars = scalars.narrow(2, output_scalars.shape[2], (scalars.shape[2] - output_scalars.shape[2]))
        output_scalars = self.scalar_act(output_scalars)
        gate_scalars = self.gate_act(gate_scalars)
        x_grid = self.so3_grid.to_grid(inputs)
        x_grid_1, x_grid_2 = torch.chunk(x_grid, chunks=2, dim=-1)
        x_grid = x_grid_1 * x_grid_2
        x_grid = self.grid_drop(x_grid)
        output_vectors = self.so3_grid.from_grid(x_grid)
        output_vectors = output_vectors * gate_scalars
        outputs = output_vectors
        outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + output_scalars
        return outputs

    def forward(self, inputs, scalars):
        return torch.utils.checkpoint.checkpoint(self.kernel, inputs, scalars, use_reentrant=False)


def get_activation(act_name, lmax, mmax, grid_resolution_list=None, use_m_primary=False):
    check_activation_name(act_name)
    if act_name == 'gate':
        act_class = GateActivation
    elif act_name == 's2':
        act_class = S2Activation
    elif act_name == 'sep_s2':
        act_class = SeparableS2Activation
    elif act_name == 's2_swiglu':
        act_class = S2Activation_SwiGLU
    elif act_name == 's2_swiglu_mem':
        act_class = S2Activation_SwiGLU_MemoryEfficient
    elif act_name == 'sep_s2_swiglu':
        act_class = SeparableS2Activation_SwiGLU
    elif act_name == 'sep-merge_s2_swiglu':
        act_class = SeparableS2Activation_SwiGLU_Merge
    elif act_name == 'sep_s2_swiglu_mem':
        act_class = SeparableS2Activation_SwiGLU_MemoryEfficient
    elif act_name == 'sep-merge_s2_swiglu_mem':
        act_class = SeparableS2Activation_SwiGLU_Merge_MemoryEfficient
    elif act_name == 'sep_s2_square':
        act_class = SeparableS2Activation_Square
    elif act_name == 'sep-merge_gates2_swiglu':
        act_class = SeparableGateS2Activation_SwiGLU_Merge
    elif act_name == 'sep-merge_gates2_swiglu_mem':
        act_class = SeparableGateS2Activation_SwiGLU_Merge_MemoryEfficient
    args = {'lmax': lmax, 'mmax': mmax, 'use_m_primary': use_m_primary}
    if act_name != 'gate':
        args['grid_resolution_list'] = grid_resolution_list
    return act_class(**args)


def add_dropout(act, drop):
    """Add an extra Dropout layer to an existing activation in-place."""
    attribute_name_list = ['act', 'gate_act', 'scalar_act']
    for attr_name in attribute_name_list:
        if attr_name == 'gate_act' and isinstance(act, SeparableGateS2Activation_SwiGLU_Merge):
            continue
        if hasattr(act, attr_name):
            temp = copy.deepcopy(getattr(act, attr_name))
            update_act_list = [temp, nn.Dropout(drop)]
            delattr(act, attr_name)
            setattr(act, attr_name, nn.Sequential(*update_act_list))
    if hasattr(act, 'grid_drop'):
        delattr(act, 'grid_drop')
        setattr(act, 'grid_drop', nn.Dropout(drop))


# ---------------------------------------------------------------------------
# Drop / softmax helpers used inside the transformer
# ---------------------------------------------------------------------------

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class GraphDropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x, batch):
        batch_size = int(batch.max().item()) + 1
        shape = (batch_size,) + (1,) * (x.ndim - 1)
        ones = torch.ones(shape, dtype=x.dtype, device=x.device)
        drop = drop_path(ones, self.drop_prob, self.training)
        return x * drop[batch]

    def extra_repr(self):
        return 'drop_prob={}'.format(self.drop_prob)


class EquivariantDropout(nn.Module):
    def __init__(self, lmax, mmax, drop_prob, use_m_primary=False):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.drop_prob = drop_prob
        self.use_m_primary = use_m_primary

        self.drop = nn.Dropout(drop_prob, True)

        expand_index = []
        if not self.use_m_primary:
            for l in range(self.lmax + 1):
                mmax_l = min(l, self.mmax)
                l_index_tensor = torch.ones(((2 * mmax_l + 1),), dtype=torch.long) * l
                expand_index.append(l_index_tensor)
        else:
            for m in range(self.mmax + 1):
                l_index = torch.arange((self.lmax + 1 - m))
                expand_index.append(l_index)
                if m > 0:
                    expand_index.append(l_index)
        expand_index = torch.cat(expand_index, dim=0).long()
        self.register_buffer('expand_index', expand_index)

    def extra_repr(self):
        return 'lmax={}, mmax={}, drop_prob={}, use_m_primary={}'.format(
            self.lmax, self.mmax, self.drop_prob, self.use_m_primary
        )

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        assert len(x.shape) == 3
        shape = (x.shape[0], (self.lmax + 1), x.shape[2])
        mask = torch.ones(shape, dtype=x.dtype, device=x.device)
        mask = self.drop(mask)
        mask = torch.index_select(mask, dim=1, index=self.expand_index)
        return x * mask


class SoftCap(nn.Module):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap

    def forward(self, inputs):
        outputs = inputs / self.cap
        outputs = nn.functional.tanh(outputs)
        return outputs * self.cap

    def __repr__(self):
        return f"{self.__class__.__name__}(cap={self.cap})"


class GraphSoftmax(nn.Module):
    """Native PyTorch port of the V3 ``GraphSoftmax`` (only the index path)."""
    def __init__(self, eps=1e-16, exp_dropout=0.0, softcap=None):
        super().__init__()
        self.eps = eps
        self.exp_dropout = exp_dropout
        self.dropout = nn.Dropout(exp_dropout) if self.exp_dropout > 0.0 else nn.Identity()
        self.softcap = SoftCap(cap=softcap) if softcap is not None else nn.Identity()

    def forward(self, src, index=None, ptr=None, num_nodes=None, dim=0, exp_rescale=None):
        if index is None:
            raise NotImplementedError("GraphSoftmax requires index in this port")
        src = self.softcap(src)

        N = num_nodes if num_nodes is not None else int(index.max().item()) + 1

        index_expand = index
        for _ in range(src.dim() - index.dim()):
            index_expand = index_expand.unsqueeze(-1)
        index_expand = index_expand.expand_as(src)

        size = list(src.shape)
        size[dim] = N

        max_value = torch.full(size, -1e10, dtype=src.dtype, device=src.device)
        max_value.scatter_reduce_(dim, index_expand, src.detach(), reduce='amax', include_self=False)

        out = src - max_value.gather(dim, index_expand)
        out = out.exp()
        if exp_rescale is not None:
            out = out * exp_rescale
        out = self.dropout(out)

        sum_value = torch.zeros(size, dtype=src.dtype, device=src.device)
        sum_value.scatter_add_(dim, index_expand, out)
        return out / (sum_value.gather(dim, index_expand) + self.eps)

    def extra_repr(self):
        return 'eps={}'.format(self.eps)


# ---------------------------------------------------------------------------
# Edge-degree embedding (input block)
# ---------------------------------------------------------------------------

class EdgeDegreeEmbedding(nn.Module):
    def __init__(self, num_channels, lmax, mmax, so3_rotation,
                 max_num_elements, edge_channels_list,
                 use_atom_edge_embedding, rescale_factor):
        super().__init__()
        self.num_channels = num_channels
        self.lmax = lmax
        self.mmax = mmax
        self.so3_rotation = so3_rotation

        self.max_num_elements = max_num_elements
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.use_atom_edge_embedding = use_atom_edge_embedding

        if self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None

        self.edge_channels_list.append((self.lmax + 1) * self.num_channels)
        self.rad_func = RadialFunction(
            self.edge_channels_list, lmax=self.lmax, mmax=self.mmax,
            use_rad_l_parametrization=True, use_expand=False,
        )

        self.rescale_factor = rescale_factor

    def forward(self, atomic_numbers, edge_distance, edge_index, edge_envelope_weight=None):
        if self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]
            target_element = atomic_numbers[edge_index[1]]
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            x_edge = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)
        else:
            x_edge = edge_distance

        x_edge_m0 = self.rad_func(x_edge)

        if edge_envelope_weight is not None:
            x_edge_m0 = x_edge_m0 * edge_envelope_weight

        x_edge_m0 = x_edge_m0.view(x_edge_m0.shape[0], (self.lmax + 1), self.num_channels)
        x_edge = torch.bmm(
            self.so3_rotation.wigner_inv.narrow(2, 0, (self.lmax + 1)),
            x_edge_m0,
        )

        outputs = reduce_edge(
            inputs=x_edge,
            edge_index=edge_index[1],
            output_shape=[atomic_numbers.shape[0], x_edge.shape[1], x_edge.shape[2]],
        )
        return outputs / self.rescale_factor

    def extra_repr(self):
        return 'rescale_factor={}'.format(self.rescale_factor)


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class EquivariantGraphAttention(nn.Module):
    def __init__(
        self,
        num_in_channels,
        num_hidden_channels,
        num_heads,
        attn_alpha_channels,
        attn_value_channels,
        num_out_channels,
        lmax,
        mmax,
        so3_rotation,
        grid_resolution_list,
        max_num_elements,
        edge_channels_list,
        use_atom_edge_embedding=True,
        activation='sep-merge_s2_swiglu',
        use_attn_renorm=True,
        use_add_merge=False,
        use_rad_l_parametrization=True,
        softcap=None,
        eps=1e-16,
        alpha_drop=0.0,
        attn_mask_rate=0.0,
        attn_weights_drop=0.0,
        value_drop=0.0,
    ):
        super().__init__()

        self.num_in_channels = num_in_channels
        self.num_hidden_channels = num_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.num_out_channels = num_out_channels
        self.lmax = lmax
        self.mmax = mmax

        self.so3_rotation = so3_rotation
        self.grid_resolution_list = grid_resolution_list

        self.max_num_elements = max_num_elements
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.use_atom_edge_embedding = use_atom_edge_embedding

        self.activation = activation
        self.use_attn_renorm = use_attn_renorm
        self.use_add_merge = use_add_merge
        self.use_rad_l_parametrization = use_rad_l_parametrization
        self.softcap = softcap
        self.eps = eps

        if self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None

        if not self.use_add_merge:
            if self.use_rad_l_parametrization:
                self.edge_channels_list.append((self.num_in_channels * (self.lmax + 1) * 2))
            else:
                num_rad_out_channels = 0
                for m in range(self.mmax + 1):
                    num_rad_out_channels = num_rad_out_channels + (self.lmax + 1 - m)
                num_rad_out_channels = num_rad_out_channels * self.num_in_channels
                num_rad_out_channels = num_rad_out_channels * 2
                self.edge_channels_list.append(num_rad_out_channels)
            self.rad_func = RadialFunction(
                self.edge_channels_list, lmax=self.lmax,
                mmax=(self.lmax if self.use_rad_l_parametrization else self.mmax),
                use_rad_l_parametrization=self.use_rad_l_parametrization,
                use_expand=True,
            )
        else:
            assert self.use_rad_l_parametrization
            self.edge_channels_list.append((self.num_in_channels * (self.lmax + 1) * 2))
            self.rad_func = RadialFunction(
                self.edge_channels_list, lmax=self.lmax, mmax=self.lmax,
                use_rad_l_parametrization=True, use_expand=True,
            )
            self.rad_func.net[-1].weight.data.mul_((1.0 / math.sqrt(2.0)))

        check_activation_name(self.activation)
        if (('swiglu' in self.activation) or ('square' in self.activation)):
            self.num_hidden_channels = self.num_hidden_channels * 2
            self.use_swiglu = True
        else:
            self.use_swiglu = False

        extra_m0_out_channels = self.num_heads * self.attn_alpha_channels
        if has_scalars(self.activation):
            self.split_m0_channels_list = [extra_m0_out_channels]
            if 'sep-merge_gates2_swiglu' in self.activation:
                temp = self.num_hidden_channels + (self.num_hidden_channels // 2)
                extra_m0_out_channels = extra_m0_out_channels + temp
                self.split_m0_channels_list.append(temp)
            elif 'sep' in self.activation:
                extra_m0_out_channels = extra_m0_out_channels + self.num_hidden_channels
                self.split_m0_channels_list.append(self.num_hidden_channels)
            elif 'gate' in self.activation:
                temp = self.lmax * self.num_hidden_channels
                extra_m0_out_channels = extra_m0_out_channels + temp
                self.split_m0_channels_list.append(temp)
            else:
                raise ValueError

        self.so2_linear_1 = SO2Linear(
            ((2 * self.num_in_channels) if not self.use_add_merge else self.num_in_channels),
            self.num_hidden_channels, self.lmax, self.mmax,
            extra_m0_out_channels=extra_m0_out_channels,
        )

        self.alpha_norm = nn.LayerNorm(self.attn_alpha_channels) if self.use_attn_renorm else nn.Identity()
        self.alpha_act = nn.SiLU() if alpha_drop != 0.0 else SmoothLeakyReLU()
        self.alpha_dropout = nn.Dropout(alpha_drop) if alpha_drop != 0.0 else nn.Identity()
        self.alpha_dot = nn.Parameter(torch.randn(self.num_heads, self.attn_alpha_channels))
        std = 1.0 / math.sqrt(self.attn_alpha_channels)
        nn.init.uniform_(self.alpha_dot, -std, std)
        self.attn_softmax = GraphSoftmax(eps=self.eps, exp_dropout=attn_mask_rate, softcap=self.softcap)
        self.attn_weights_dropout = nn.Dropout(attn_weights_drop) if attn_weights_drop != 0.0 else nn.Identity()

        self.act = get_activation(
            act_name=self.activation, lmax=self.lmax, mmax=self.mmax,
            grid_resolution_list=self.grid_resolution_list, use_m_primary=True,
        )
        if value_drop != 0.0:
            add_dropout(self.act, value_drop)

        self.so2_linear_2 = SO2Linear(
            (self.num_hidden_channels if not self.use_swiglu else (self.num_hidden_channels // 2)),
            self.num_heads * self.attn_value_channels, self.lmax, self.mmax,
            extra_m0_out_channels=None,
        )
        if '-merge' in self.activation:
            temp = self.so2_linear_2.num_in_channels
            self.so2_linear_2.fc_m0.weight.data[0:temp, :].mul_(1.0 / math.sqrt(2.0))

        self.proj = SO3Linear(self.num_heads * self.attn_value_channels, self.num_out_channels, lmax=self.lmax)

    def forward(self, x, source_atomic_numbers, target_atomic_numbers,
                edge_distance, edge_index, edge_envelope_weight=None):
        num_nodes = x.shape[0]

        if self.use_atom_edge_embedding:
            source_embedding = self.source_embedding(source_atomic_numbers)
            target_embedding = self.target_embedding(target_atomic_numbers)
            x_edge = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)
        else:
            x_edge = edge_distance

        x_edge_weight = self.rad_func(x_edge)

        x = x.to(x_edge_weight.dtype)
        x_source = torch.index_select(x, index=edge_index[0], dim=0)
        x_target = torch.index_select(x, index=edge_index[1], dim=0)
        if not self.use_add_merge:
            x_message = torch.cat((x_source, x_target), dim=2)
            if self.use_rad_l_parametrization:
                x_message = x_message * x_edge_weight
                x_message = self.so3_rotation.rotate(x_message)
            else:
                x_message = self.so3_rotation.rotate(x_message)
                x_message = x_message * x_edge_weight
        else:
            x_edge_weight_source = x_edge_weight.narrow(2, 0, self.num_in_channels)
            x_edge_weight_target = x_edge_weight.narrow(2, self.num_in_channels, self.num_in_channels)
            x_source = x_source * x_edge_weight_source
            x_target = x_target * x_edge_weight_target
            x_message = x_source + x_target
            x_message = self.so3_rotation.rotate(x_message)

        x_message, x_m0_extra = self.so2_linear_1(x_message)

        if has_scalars(self.activation):
            x_alpha = x_m0_extra.narrow(1, 0, self.split_m0_channels_list[0])
            x_scalar = x_m0_extra.narrow(1, self.split_m0_channels_list[0], self.split_m0_channels_list[1])
        else:
            x_alpha = x_m0_extra
            x_scalar = None
        act_input_dict = prepare_activation_forward_param(
            act_name=self.activation, inputs=x_message, scalars=x_scalar,
        )
        x_message = self.act(**act_input_dict)

        x_message = self.so2_linear_2(x_message)

        x_alpha = x_alpha.view(-1, self.num_heads, self.attn_alpha_channels)
        x_alpha = self.alpha_norm(x_alpha)
        x_alpha = self.alpha_act(x_alpha)
        x_alpha = self.alpha_dropout(x_alpha)
        alpha = torch.einsum('bik, ik -> bi', x_alpha, self.alpha_dot)
        alpha = self.attn_softmax(alpha, edge_index[1], num_nodes=num_nodes, exp_rescale=edge_envelope_weight)
        if edge_envelope_weight is not None:
            alpha = alpha * edge_envelope_weight
        alpha = alpha.view(alpha.shape[0], 1, self.num_heads, 1)
        alpha = self.attn_weights_dropout(alpha)
        if torch.is_autocast_enabled():
            alpha = alpha.to(torch.float16)

        attn = x_message
        attn = attn.view(attn.shape[0], attn.shape[1], self.num_heads, self.attn_value_channels)
        attn = attn * alpha
        attn = attn.view(attn.shape[0], attn.shape[1], self.num_heads * self.attn_value_channels)
        x_message = attn

        x_message = self.so3_rotation.rotate_inv(x_message)

        x_message = reduce_edge(
            inputs=x_message, edge_index=edge_index[1],
            output_shape=(num_nodes, x_message.shape[1], x_message.shape[2]),
        )

        return self.proj(x_message)


class GatedSwiGLUGridMLP(nn.Module):
    def __init__(self, num_in_channels, num_hidden_channels, dropout):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.num_hidden_channels = num_hidden_channels
        self.dropout = dropout

        self.gating_linear = nn.Linear(self.num_in_channels, self.num_hidden_channels)
        self.gate_act = nn.Sigmoid()
        self.grid_linear_1 = nn.Linear(self.num_hidden_channels, 2 * self.num_hidden_channels, bias=False)
        self.grid_drop = (nn.Dropout(self.dropout) if self.dropout > 0.0 else nn.Identity())
        self.grid_linear_2 = nn.Linear(self.num_hidden_channels, self.num_hidden_channels, bias=False)

    def forward(self, input_grid, scalars):
        gate_scalars = self.gating_linear(scalars)
        gate_scalars = self.gate_act(gate_scalars)
        output_grid = self.grid_linear_1(input_grid)
        output_grid_1, output_grid_2 = torch.chunk(output_grid, chunks=2, dim=-1)
        output_grid_1 = output_grid_1 * gate_scalars
        output_grid = output_grid_1 * output_grid_2
        output_grid = self.grid_drop(output_grid)
        return self.grid_linear_2(output_grid)


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        num_in_channels,
        num_hidden_channels,
        num_out_channels,
        lmax,
        mmax,
        grid_resolution_list,
        activation='sep-merge_s2_swiglu',
        use_grid_mlp=True,
        dropout=0.0,
    ):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.num_hidden_channels = num_hidden_channels
        self.num_out_channels = num_out_channels
        self.lmax = lmax
        self.mmax = mmax
        self.grid_resolution_list = grid_resolution_list
        self.activation = activation
        self.use_grid_mlp = use_grid_mlp

        check_activation_name(self.activation)
        if self.use_grid_mlp:
            assert 's2' in self.activation

        self.so3_linear_1 = SO3Linear(self.num_in_channels, self.num_hidden_channels, lmax=self.lmax)

        if self.use_grid_mlp:
            if 'sep' in self.activation:
                if 'swiglu' in self.activation:
                    self.scalar_mlp = nn.Sequential(
                        LinearSwiGLU(self.num_in_channels, self.num_hidden_channels),
                        (nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()),
                    )
                elif 'square' in self.activation:
                    self.scalar_mlp = nn.Sequential(
                        LinearSquare(self.num_in_channels, self.num_hidden_channels),
                        (nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()),
                    )
                else:
                    self.scalar_mlp = nn.Sequential(
                        nn.Linear(self.num_in_channels, self.num_hidden_channels),
                        nn.SiLU(),
                        (nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()),
                    )
            else:
                self.scalar_mlp = None

            self.so3_grid = SO3Grid(
                lmax=self.lmax, mmax=self.lmax,
                resolution_list=self.grid_resolution_list, use_m_primary=False,
            )

            if 'swiglu' in self.activation:
                if 'gates2' not in self.activation:
                    self.grid_mlp = nn.Sequential(
                        LinearSwiGLU(self.num_hidden_channels, self.num_hidden_channels, bias=False),
                        (nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()),
                        nn.Linear(self.num_hidden_channels, self.num_hidden_channels, bias=False),
                    )
                else:
                    self.grid_mlp = GatedSwiGLUGridMLP(self.num_in_channels, self.num_hidden_channels, dropout)
            elif 'square' in self.activation:
                self.grid_mlp = nn.Sequential(
                    LinearSquare(self.num_hidden_channels, self.num_hidden_channels, bias=False),
                    (nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()),
                    nn.Linear(self.num_hidden_channels, self.num_hidden_channels, bias=False),
                )
            else:
                self.grid_mlp = nn.Sequential(
                    nn.Linear(self.num_hidden_channels, self.num_hidden_channels, bias=False),
                    nn.SiLU(),
                    (nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()),
                    nn.Linear(self.num_hidden_channels, self.num_hidden_channels, bias=False),
                    nn.SiLU(),
                    (nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()),
                    nn.Linear(self.num_hidden_channels, self.num_hidden_channels, bias=False),
                )
        else:
            assert self.activation in ['gate', 's2', 'sep_s2', 'sep-merge_gates2_swiglu']
            if self.activation == 'gate':
                self.gating_linear = nn.Linear(self.num_in_channels, (self.lmax * self.num_hidden_channels))
            elif self.activation == 'sep_s2':
                self.gating_linear = nn.Linear(self.num_in_channels, self.num_hidden_channels)
            elif self.activation == 'sep-merge_gates2_swiglu':
                del self.so3_linear_1
                self.so3_linear_1 = SO3Linear(self.num_in_channels, 2 * self.num_hidden_channels, lmax=self.lmax)
                self.gating_linear = nn.Linear(
                    self.num_in_channels,
                    (2 * self.num_hidden_channels + self.num_hidden_channels),
                )
            else:
                self.gating_linear = None
            self.act = get_activation(
                act_name=self.activation, lmax=self.lmax, mmax=self.lmax,
                grid_resolution_list=self.grid_resolution_list, use_m_primary=False,
            )

            if dropout != 0.0:
                add_dropout(self.act, dropout)

        self.so3_linear_2 = SO3Linear(self.num_hidden_channels, self.num_out_channels, lmax=self.lmax)
        if '-merge' in self.activation:
            self.so3_linear_2.weight.data[0, :, :].mul_(1.0 / math.sqrt(2.0))

    def forward(self, inputs):
        gating_scalars = None
        if self.use_grid_mlp:
            if self.scalar_mlp is not None:
                gating_scalars = self.scalar_mlp(inputs.narrow(1, 0, 1))
        else:
            if self.gating_linear is not None:
                gating_scalars = self.gating_linear(inputs.narrow(1, 0, 1))

        outputs = self.so3_linear_1(inputs)

        if self.use_grid_mlp:
            output_grid = self.so3_grid.to_grid(outputs)
            if 'gates2' not in self.activation:
                if '_mem' not in self.activation:
                    output_grid = self.grid_mlp(output_grid)
                else:
                    output_grid = torch.utils.checkpoint.checkpoint(self.grid_mlp, output_grid)
            else:
                if '_mem' not in self.activation:
                    output_grid = self.grid_mlp(output_grid, inputs.narrow(1, 0, 1))
                else:
                    output_grid = torch.utils.checkpoint.checkpoint(
                        self.grid_mlp, output_grid, inputs.narrow(1, 0, 1)
                    )
            outputs = self.so3_grid.from_grid(output_grid)

            if self.scalar_mlp is not None:
                if '-merge' not in self.activation:
                    outputs = torch.cat(
                        (gating_scalars, outputs.narrow(1, 1, outputs.shape[1] - 1)), dim=1
                    )
                else:
                    outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + gating_scalars
        else:
            act_input_dict = prepare_activation_forward_param(
                act_name=self.activation, inputs=outputs, scalars=gating_scalars,
            )
            outputs = self.act(**act_input_dict)

        return self.so3_linear_2(outputs)


class TransBlockV3(nn.Module):
    def __init__(
        self,
        num_in_channels,
        attn_hidden_channels,
        num_heads,
        attn_alpha_channels,
        attn_value_channels,
        ffn_hidden_channels,
        num_out_channels,
        lmax,
        mmax,
        so3_rotation,
        attn_grid_resolution_list,
        ffn_grid_resolution_list,
        max_num_elements,
        edge_channels_list,
        use_atom_edge_embedding=True,
        attn_activation='sep-merge_s2_swiglu',
        use_attn_renorm=True,
        use_add_merge=False,
        use_rad_l_parametrization=True,
        softcap=None,
        attn_eps=1e-16,
        ffn_activation='sep-merge_s2_swiglu',
        use_grid_mlp=True,
        norm_type='sep_layer_norm',
        alpha_drop=0.0,
        attn_mask_rate=0.0,
        attn_weights_drop=0.0,
        value_drop=0.0,
        drop_path_rate=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
    ):
        super().__init__()

        self.norm_1 = get_normalization_layer(norm_type, lmax=lmax, num_channels=num_in_channels)

        self.ga = EquivariantGraphAttention(
            num_in_channels=num_in_channels,
            num_hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            num_out_channels=num_in_channels,
            lmax=lmax, mmax=mmax,
            so3_rotation=so3_rotation,
            grid_resolution_list=attn_grid_resolution_list,
            max_num_elements=max_num_elements,
            edge_channels_list=edge_channels_list,
            use_atom_edge_embedding=use_atom_edge_embedding,
            activation=attn_activation,
            use_attn_renorm=use_attn_renorm,
            use_add_merge=use_add_merge,
            use_rad_l_parametrization=use_rad_l_parametrization,
            softcap=softcap,
            eps=attn_eps,
            alpha_drop=alpha_drop,
            attn_mask_rate=attn_mask_rate,
            attn_weights_drop=attn_weights_drop,
            value_drop=value_drop,
        )

        if 'rms_norm' in norm_type:
            if self.ga.alpha_norm is not None:
                del self.ga.alpha_norm
                self.ga.alpha_norm = RMSNorm(attn_alpha_channels)

        self.drop_path = GraphDropPath(drop_path_rate)
        self.proj_drop = EquivariantDropout(lmax=lmax, mmax=lmax, drop_prob=proj_drop) if proj_drop > 0.0 else None

        self.norm_2 = get_normalization_layer(norm_type, lmax=lmax, num_channels=num_in_channels)

        self.ffn = FeedForwardNetwork(
            num_in_channels=num_in_channels,
            num_hidden_channels=ffn_hidden_channels,
            num_out_channels=num_out_channels,
            lmax=lmax, mmax=mmax,
            grid_resolution_list=ffn_grid_resolution_list,
            activation=ffn_activation,
            use_grid_mlp=use_grid_mlp,
            dropout=ffn_drop,
        )

        if num_in_channels != num_out_channels:
            self.ffn_shortcut = SO3Linear(num_in_channels, num_out_channels, lmax=lmax)
        else:
            self.ffn_shortcut = None

    def forward(self, x, source_atomic_numbers, target_atomic_numbers,
                edge_distance, edge_index, edge_envelope_weight=None, batch=None):
        outputs = x
        x_res = x

        outputs = self.norm_1(outputs)
        outputs = self.ga(
            outputs, source_atomic_numbers, target_atomic_numbers,
            edge_distance, edge_index, edge_envelope_weight,
        )

        if self.drop_path is not None:
            outputs = self.drop_path(outputs, batch)
        if self.proj_drop is not None:
            outputs = self.proj_drop(outputs)

        outputs = outputs + x_res

        x_res = outputs
        outputs = self.norm_2(outputs)
        outputs = self.ffn(outputs)

        if self.drop_path is not None:
            outputs = self.drop_path(outputs, batch)
        if self.proj_drop is not None:
            outputs = self.proj_drop(outputs)

        if self.ffn_shortcut is not None:
            x_res = self.ffn_shortcut(x_res)

        return outputs + x_res


# ---------------------------------------------------------------------------
# Output blocks
# ---------------------------------------------------------------------------

def get_stress_cg_change_mat() -> torch.Tensor:
    change_mat = torch.tensor([
        [3 ** (-0.5), 0, 0, 0, 3 ** (-0.5), 0, 0, 0, 3 ** (-0.5)],
        [0, 0, 0, 0, 0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0],
        [0, 0, -(2 ** (-0.5)), 0, 0, 0, 2 ** (-0.5), 0, 0],
        [0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0, 0, 0, 0, 0],
        [0, 0, 0.5 ** 0.5, 0, 0, 0, 0.5 ** 0.5, 0, 0],
        [0, 2 ** (-0.5), 0, 2 ** (-0.5), 0, 0, 0, 0, 0],
        [-(6 ** (-0.5)), 0, 0, 0, 2 * 6 ** (-0.5), 0, 0, 0, -(6 ** (-0.5))],
        [0, 0, 0, 0, 0, 2 ** (-0.5), 0, 2 ** (-0.5), 0],
        [-(2 ** (-0.5)), 0, 0, 0, 0, 0, 0, 0, 2 ** (-0.5)],
    ])
    return change_mat


class ScalarFeedForwardNetwork(nn.Module):
    """Plain MLP head used for energy prediction in V3."""
    def __init__(self, num_in_channels, num_hidden_channels, num_out_channels, dropout=0.0):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.num_hidden_channels = num_hidden_channels
        self.num_out_channels = num_out_channels
        self.linear_1 = nn.Linear(self.num_in_channels, self.num_hidden_channels, bias=True)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.linear_2 = nn.Linear(self.num_hidden_channels, self.num_out_channels, bias=True)

    def forward(self, inputs):
        outputs = self.linear_1(inputs)
        outputs = self.act(outputs)
        outputs = self.dropout(outputs)
        return self.linear_2(outputs)


class EquivariantGraphAttentionStressHead(EquivariantGraphAttention):
    """Direct stress head from V3 (kept for completeness; unused by default)."""
    def __init__(
        self,
        num_in_channels, num_hidden_channels, num_heads,
        attn_alpha_channels, attn_value_channels, num_out_channels,
        lmax, mmax, so3_rotation, grid_resolution_list,
        max_num_elements, edge_channels_list,
        use_atom_edge_embedding=True, activation='sep-merge_s2_swiglu',
        use_attn_renorm=True, use_add_merge=False,
        use_rad_l_parametrization=True, softcap=None,
        alpha_drop=0.0, attn_mask_rate=0.0,
        attn_weights_drop=0.0, value_drop=0.0,
    ):
        assert num_out_channels == 1
        super().__init__(
            num_in_channels, num_hidden_channels, num_heads,
            attn_alpha_channels, attn_value_channels, num_out_channels,
            lmax, mmax, so3_rotation, grid_resolution_list,
            max_num_elements, edge_channels_list, use_atom_edge_embedding,
            activation, use_attn_renorm, use_add_merge,
            use_rad_l_parametrization, softcap,
            1e-16, alpha_drop, attn_mask_rate, attn_weights_drop, value_drop,
        )
        zero_padded_change_matrix = get_stress_cg_change_mat()
        zero_padded_change_matrix[1:4, :] = 0.0
        self.register_buffer('zero_padded_change_matrix', zero_padded_change_matrix)

    def forward(self, x, source_atomic_numbers, target_atomic_numbers,
                edge_distance, edge_index, edge_envelope_weight=None,
                batch_size=None, batch=None):
        outputs = super().forward(
            x, source_atomic_numbers, target_atomic_numbers,
            edge_distance, edge_index, edge_envelope_weight,
        )
        outputs = outputs.view(outputs.shape[0], ((self.lmax + 1) ** 2))

        stress = scatter_mean(
            src=outputs.narrow(1, 0, 9), index=batch, dim=0, dim_size=batch_size,
        )
        stress = torch.einsum('ni, ij -> nj', stress, self.zero_padded_change_matrix)
        return stress


class FeedForwardNetworkStressHead(FeedForwardNetwork):
    """Direct stress head built from a feed-forward network (unused by default)."""
    def __init__(self, num_in_channels, num_hidden_channels, num_out_channels,
                 lmax, mmax, grid_resolution_list,
                 activation='sep-merge_s2_swiglu', use_grid_mlp=True, dropout=0.0):
        assert num_out_channels == 1
        super().__init__(
            num_in_channels, num_hidden_channels, num_out_channels,
            lmax, mmax, grid_resolution_list,
            activation, use_grid_mlp, dropout,
        )
        zero_padded_change_matrix = get_stress_cg_change_mat()
        zero_padded_change_matrix[1:4, :] = 0.0
        self.register_buffer('zero_padded_change_matrix', zero_padded_change_matrix)

    def forward(self, x, batch_size, batch):
        outputs = super().forward(x)
        outputs = outputs.view(outputs.shape[0], ((self.lmax + 1) ** 2))
        stress = scatter_mean(
            src=outputs.narrow(1, 0, 9), index=batch, dim=0, dim_size=batch_size,
        )
        stress = torch.einsum('ni, ij -> nj', stress, self.zero_padded_change_matrix)
        return stress


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class EquiformerV3(nn.Module):
    """
    EquiformerV3 wrapped to take an :class:`AtomsData` and return one with the
    predicted ``energy`` / ``forces`` / ``stress`` / ``virial`` populated.

    Forces, virial and stress are obtained via :class:`GradientOutput`
    (autograd through ``edge_vectors`` / displacement), matching the way
    ``equiformerV2.py`` integrates with the IANN trainer/calculator.
    """

    def __init__(
        self,
        device='cpu',
        num_channels=128,
        num_layers=12,
        norm_data=False,
        norm_per_atom=False,
        data_stddev=1.0,
        data_mean=0.0,
        **kwargs,
    ):
        super().__init__()

        self.random_seed: int = kwargs.get('random_seed', 666)
        torch.manual_seed(self.random_seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(self.random_seed)

        # Basic / structural settings ---------------------------------------
        self.cutoff: float = kwargs.get('cutoff', 5.5)
        self.compute_forces = kwargs.get('compute_forces', False)
        self.compute_stress = kwargs.get('compute_stress', False)
        self.compute_virial = kwargs.get('compute_virial', False)

        self.species = kwargs.get('species', None)
        self.universal_elems = kwargs.get('universal_elems', True)
        if self.universal_elems:
            self.max_num_elements = 119
        else:
            if self.species is not None:
                from ase.data import atomic_numbers
                self.max_num_elements = len(self.species)
            else:
                self.max_num_elements = 119
        # Allow explicit override (mirrors V3 default of 128)
        self.max_num_elements = kwargs.get('max_num_elements', self.max_num_elements)

        element_mapping = torch.arange(120).long()
        if not self.universal_elems and self.species is not None:
            from ase.data import atomic_numbers
            Zs = torch.tensor([atomic_numbers[s] for s in self.species], dtype=torch.long)
            indices = torch.arange(len(Zs)).long()
            element_mapping.index_copy_(0, Zs, indices)
        self.register_buffer('element_mapping', element_mapping, persistent=False)

        self.device = torch.device(device)
        self.dtype = torch.float32

        # V3 hyper-parameters (single lmax/mmax) ----------------------------
        self.lmax: int = int(kwargs.get('lmax', 6))
        self.mmax: int = int(kwargs.get('mmax', 2))

        self.num_layers = num_layers
        self.num_channels = num_channels
        self.attn_hidden_channels = kwargs.get('attn_hidden_channels', num_channels)
        self.num_heads = kwargs.get('num_heads', 8)
        self.attn_alpha_channels = kwargs.get('attn_alpha_channels', max(num_channels // 4, 1))
        self.attn_value_channels = kwargs.get('attn_value_channels', max(num_channels // 8, 1))
        self.ffn_hidden_channels = kwargs.get('ffn_hidden_channels', num_channels)

        self.attn_grid_resolution_list = kwargs.get('attn_grid_resolution_list', [20, 8])
        self.ffn_grid_resolution_list = kwargs.get('ffn_grid_resolution_list', [20, 20])

        self.norm_type = kwargs.get('norm_type', 'merge_layer_norm')

        self.edge_channels = kwargs.get('edge_channels', num_channels)
        self.use_atom_edge_embedding = kwargs.get('use_atom_edge_embedding', True)
        self.use_envelope = kwargs.get('use_envelope', True)

        self.attn_activation = kwargs.get('attn_activation', 'sep-merge_gates2_swiglu')
        self.use_attn_renorm = kwargs.get('use_attn_renorm', True)
        self.use_add_merge = kwargs.get('use_add_merge', False)
        self.use_rad_l_parametrization = kwargs.get('use_rad_l_parametrization', True)
        self.softcap = kwargs.get('softcap', None)
        self.attn_eps = kwargs.get('attn_eps', 1e-16)
        self.ffn_activation = kwargs.get('ffn_activation', 'sep-merge_gates2_swiglu')
        self.use_grid_mlp = kwargs.get('use_grid_mlp', True)

        self.alpha_drop = kwargs.get('alpha_drop', 0.0)
        self.attn_mask_rate = kwargs.get('attn_mask_rate', 0.0)
        self.attn_weights_drop = kwargs.get('attn_weights_drop', 0.1)
        self.value_drop = kwargs.get('value_drop', 0.0)
        self.drop_path_rate = kwargs.get('drop_path_rate', 0.05)
        self.proj_drop = kwargs.get('proj_drop', 0.0)
        self.ffn_drop = kwargs.get('ffn_drop', 0.0)

        # Direct prediction is supported in code but the autograd pipeline is
        # the active path; force/stress heads are off by default.
        self.direct_prediction = kwargs.get('direct_prediction', False)
        self.use_direct_force_head = kwargs.get('use_direct_force_head', False)
        self.use_direct_stress_head = kwargs.get('use_direct_stress_head', False)
        self.use_gate_force_head = kwargs.get('use_gate_force_head', True)

        self.num_radial_basis = kwargs.get('num_radial_basis', 300)

        self.gradient_checkpointing_block_list = kwargs.get('gradient_checkpointing_block_list', None)
        if self.gradient_checkpointing_block_list is not None:
            assert len(self.gradient_checkpointing_block_list) == self.num_layers
        else:
            self.gradient_checkpointing_block_list = [0] * self.num_layers

        self.avg_num_nodes = kwargs.get('avg_num_nodes', _AVG_NUM_NODES)
        self.avg_degree = kwargs.get('avg_degree', _AVG_DEGREE)

        # Embeddings / radial basis -----------------------------------------
        self.sphere_embedding = nn.Embedding(self.max_num_elements, self.num_channels)

        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, self.num_radial_basis, 2.0)
        edge_input_channels = int(self.distance_expansion.num_output)

        self.edge_channels_list = [edge_input_channels] + [self.edge_channels] * 2

        self.envelope_func = PolynomialEnvelope(cutoff=self.cutoff, exponent=5) if self.use_envelope else None

        # Wigner-D rotation; using rotation mask only matters for the
        # gradient-prediction force path of upstream V3 (we use autograd
        # through edge_vectors here, so keeping it disabled is fine).
        self.so3_rotation = SO3Rotation(
            self.lmax, self.mmax,
            use_rotation_mask=(not self.direct_prediction),
        )

        self.edge_degree_embedding = EdgeDegreeEmbedding(
            num_channels=self.num_channels, lmax=self.lmax, mmax=self.mmax,
            so3_rotation=self.so3_rotation,
            max_num_elements=self.max_num_elements,
            edge_channels_list=self.edge_channels_list,
            use_atom_edge_embedding=self.use_atom_edge_embedding,
            rescale_factor=self.avg_degree,
        )

        # Transformer blocks -------------------------------------------------
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV3(
                num_in_channels=self.num_channels,
                attn_hidden_channels=self.attn_hidden_channels,
                num_heads=self.num_heads,
                attn_alpha_channels=self.attn_alpha_channels,
                attn_value_channels=self.attn_value_channels,
                ffn_hidden_channels=self.ffn_hidden_channels,
                num_out_channels=self.num_channels,
                lmax=self.lmax, mmax=self.mmax,
                so3_rotation=self.so3_rotation,
                attn_grid_resolution_list=self.attn_grid_resolution_list,
                ffn_grid_resolution_list=self.ffn_grid_resolution_list,
                max_num_elements=self.max_num_elements,
                edge_channels_list=self.edge_channels_list,
                use_atom_edge_embedding=self.use_atom_edge_embedding,
                attn_activation=self.attn_activation,
                use_attn_renorm=self.use_attn_renorm,
                use_add_merge=self.use_add_merge,
                use_rad_l_parametrization=self.use_rad_l_parametrization,
                softcap=self.softcap,
                attn_eps=self.attn_eps,
                ffn_activation=self.ffn_activation,
                use_grid_mlp=self.use_grid_mlp,
                norm_type=self.norm_type,
                alpha_drop=self.alpha_drop,
                attn_mask_rate=self.attn_mask_rate,
                attn_weights_drop=self.attn_weights_drop,
                value_drop=self.value_drop,
                drop_path_rate=self.drop_path_rate,
                proj_drop=self.proj_drop,
                ffn_drop=self.ffn_drop,
            )
            self.blocks.append(block)

        # Output heads -------------------------------------------------------
        self.norm = get_normalization_layer(self.norm_type, lmax=self.lmax, num_channels=self.num_channels)
        self.energy_block = ScalarFeedForwardNetwork(
            num_in_channels=self.num_channels,
            num_hidden_channels=self.ffn_hidden_channels,
            num_out_channels=1, dropout=0.0,
        )

        # Direct heads kept in code but unused unless explicitly enabled.
        self.force_block: Optional[EquivariantGraphAttention] = None
        if self.use_direct_force_head:
            self.force_block = EquivariantGraphAttention(
                num_in_channels=self.num_channels,
                num_hidden_channels=self.attn_hidden_channels,
                num_heads=self.num_heads,
                attn_alpha_channels=self.attn_alpha_channels,
                attn_value_channels=self.attn_value_channels,
                num_out_channels=1,
                lmax=self.lmax, mmax=self.mmax,
                so3_rotation=self.so3_rotation,
                grid_resolution_list=self.attn_grid_resolution_list,
                max_num_elements=self.max_num_elements,
                edge_channels_list=self.edge_channels_list,
                use_atom_edge_embedding=self.use_atom_edge_embedding,
                activation=('sep_s2' if not self.use_gate_force_head else 'gate'),
                use_attn_renorm=self.use_attn_renorm,
                use_add_merge=self.use_add_merge,
                use_rad_l_parametrization=self.use_rad_l_parametrization,
                softcap=self.softcap,
                eps=self.attn_eps,
                alpha_drop=0.0, attn_mask_rate=0.0,
                attn_weights_drop=0.0, value_drop=0.0,
            )
            if 'rms_norm' in self.norm_type:
                if self.force_block.alpha_norm is not None:
                    del self.force_block.alpha_norm
                    self.force_block.alpha_norm = RMSNorm(self.attn_alpha_channels)

        self.stress_block: Optional[FeedForwardNetworkStressHead] = None
        if self.use_direct_stress_head:
            self.stress_block = FeedForwardNetworkStressHead(
                num_in_channels=self.num_channels,
                num_hidden_channels=self.ffn_hidden_channels,
                num_out_channels=1,
                lmax=self.lmax, mmax=self.mmax,
                grid_resolution_list=self.ffn_grid_resolution_list,
                activation='gate',
                use_grid_mlp=False, dropout=0.0,
            )

        # Weight init
        self.apply(self._init_weights)

        # Normalization / shift parameters (mirrors V2)
        self.norm_data = nn.Parameter(torch.tensor(norm_data), requires_grad=False)
        self.norm_per_atom = nn.Parameter(torch.tensor(norm_per_atom), requires_grad=False)
        self.data_stddev = nn.Parameter(torch.tensor(data_stddev), requires_grad=False)
        self.data_mean = nn.Parameter(torch.tensor(data_mean), requires_grad=False)

        # Gradient pipeline for forces / virial / stress (autograd) ---------
        outputs = []
        if self.compute_forces:
            outputs.append('forces')
        if self.compute_stress:
            outputs.append('stress')
        if self.compute_virial:
            outputs.append('virial')

        if outputs:
            self.gradient_output = GradientOutput(model_outputs=outputs)
        else:
            self.gradient_output = None

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    def _init_weights(self, m):
        if (isinstance(m, nn.Linear) or isinstance(m, SO3Linear)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, RadialFunction):
            m.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, m):
        if isinstance(m, nn.Linear):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            nn.init.uniform_(m.weight, -std, std)

    def _apply_displacement(self, data: AtomsData) -> AtomsData:
        image_indices = data.image_indices
        if image_indices is None:
            return data
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

    def to(self, device):
        super().to(device)
        self.device = torch.device(device)
        return self

    # -------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------
    def forward(self, data: AtomsData) -> AtomsData:
        if self.compute_stress or self.compute_virial:
            data = self._apply_displacement(data)

        if self.universal_elems:
            atomic_numbers = data.atomic_numbers.long()
        else:
            atomic_numbers = self.element_mapping[data.atomic_numbers.long()]
        edge_index = data.edge_indices.transpose(0, 1)  # [2, E]
        edge_vectors = data.edge_vectors
        image_indices = data.image_indices
        num_atoms = atomic_numbers.shape[0]

        edge_dist = torch.norm(edge_vectors, dim=1, dtype=torch.float32)

        # Edge processing: rotation, envelope, radial expansion -------------
        edge_rot_mat = init_edge_rot_mat(edge_vectors, use_rotation_mask=(not self.direct_prediction))
        self.so3_rotation.set_wigner(edge_rot_mat)

        edge_envelope_weight = self.envelope_func(edge_dist) if self.envelope_func is not None else None
        edge_dist_expanded = self.distance_expansion(edge_dist).to(torch.float32)

        # Node embedding ----------------------------------------------------
        x = torch.zeros(
            (num_atoms, ((self.lmax + 1) ** 2), self.num_channels),
            device=edge_vectors.device, dtype=torch.float32,
        )
        x[:, 0, :] = self.sphere_embedding(atomic_numbers).to(torch.float32)

        edge_degree = self.edge_degree_embedding(
            atomic_numbers, edge_dist_expanded, edge_index, edge_envelope_weight
        )
        x = x + edge_degree

        # Transformer blocks -----------------------------------------------
        if image_indices is None:
            image_indices = torch.zeros_like(atomic_numbers, dtype=torch.long)

        source_atomic_numbers = atomic_numbers[edge_index[0]]
        target_atomic_numbers = atomic_numbers[edge_index[1]]
        for i in range(self.num_layers):
            if self.gradient_checkpointing_block_list[i] == 0:
                x = self.blocks[i](
                    x, source_atomic_numbers, target_atomic_numbers,
                    edge_dist_expanded, edge_index, edge_envelope_weight,
                    image_indices,
                )
            else:
                x = torch.utils.checkpoint.checkpoint(
                    self.blocks[i], x, source_atomic_numbers, target_atomic_numbers,
                    edge_dist_expanded, edge_index, edge_envelope_weight,
                    image_indices, use_reentrant=False,
                )

        x = self.norm(x)

        # Energy head --------------------------------------------------------
        x_scalar = x.narrow(1, 0, 1).view(num_atoms, self.num_channels)
        node_energy = self.energy_block(x_scalar).view(-1)

        atomic_energy = node_energy
        data = replace_properties(data, atomic_energy=atomic_energy)

        energy = torch.zeros(
            data.num_atoms.shape[0], device=node_energy.device, dtype=torch.float32
        )
        energy.index_add_(0, image_indices, atomic_energy)
        data = replace_properties(data, energy=energy)

        if self.norm_data:
            opt_energy = data.energy
            if opt_energy is not None:
                normalizer = self.data_stddev
                energy = normalizer * opt_energy
                if self.norm_per_atom:
                    mean_shift = data.num_atoms.to(energy.dtype) * self.data_mean
                else:
                    mean_shift = self.data_mean
                energy = energy + mean_shift
                data = replace_properties(data, energy=energy)

        if torch.isnan(energy).any():
            print("[WARNING] NaN detected in energy in EquiformerV3 forward!")
        if torch.isinf(energy).any():
            print("[WARNING] Inf detected in energy in EquiformerV3 forward!")

        # Forces / virial / stress via autograd ----------------------------
        if getattr(self, 'gradient_output', None) is not None:
            data = self.gradient_output(data)

        return data


# ---------------------------------------------------------------------------
# GradientOutput (copy of equiformerV2.GradientOutput so V3 stays self-contained)
# ---------------------------------------------------------------------------

class GradientOutput(nn.Module):
    def __init__(
        self,
        grad_on_edge_diff: bool = True,
        grad_on_positions: bool = False,
        model_outputs: List[str] = ['forces'],
        update_callback: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.grad_on_edge_diff = grad_on_edge_diff
        self.grad_on_positions = grad_on_positions
        self.update_callback = update_callback
        self.model_outputs = model_outputs

    def update_model_outputs(self, outputs: Union[List[str], str]):
        if isinstance(outputs, str):
            self.model_outputs.append(outputs)
        else:
            self.model_outputs.extend(outputs)
        if self.update_callback:
            self.update_callback()

    def forward(self, data: AtomsData, training: bool = True) -> AtomsData:
        if self.grad_on_edge_diff:
            energy = data.energy
            edge_vectors = data.edge_vectors
            forces_dim = int(torch.sum(data.num_atoms))
            edge_indices = data.edge_indices
            assert energy is not None

            outputs_list = torch.jit.annotate(List[torch.Tensor], [energy])
            inputs_list = torch.jit.annotate(List[torch.Tensor], [])
            grad_outputs_list = torch.jit.annotate(
                Optional[List[Optional[torch.Tensor]]],
                [torch.ones_like(energy, dtype=torch.float32)],
            )

            compute_forces = 'forces' in self.model_outputs
            compute_virial = 'virial' in self.model_outputs
            compute_stress = 'stress' in self.model_outputs

            if compute_forces:
                inputs_list.append(edge_vectors)

            displacement = data.displacement
            if displacement is not None and (compute_virial or compute_stress):
                inputs_list.append(displacement)

            if len(inputs_list) > 0:
                grads = torch.autograd.grad(
                    outputs=outputs_list,
                    inputs=inputs_list,
                    grad_outputs=grad_outputs_list,
                    retain_graph=training,
                    create_graph=training,
                    allow_unused=True,
                )

                idx = 0
                if compute_forces:
                    dE_ddiff = grads[idx]
                    idx += 1
                    dE_ddiff = torch.zeros_like(data.positions) if dE_ddiff is None else dE_ddiff
                    assert dE_ddiff is not None

                    i_forces = torch.zeros((forces_dim, 3), device=edge_vectors.device, dtype=torch.float32)
                    j_forces = torch.zeros_like(i_forces)
                    i_forces.index_add_(0, edge_indices[:, 0], dE_ddiff)
                    j_forces.index_add_(0, edge_indices[:, 1], -dE_ddiff)
                    forces = i_forces + j_forces
                    data = replace_properties(data, forces=forces)

                if displacement is not None and (compute_virial or compute_stress):
                    dE_ddisp = grads[idx]
                    idx += 1
                    if dE_ddisp is not None:
                        virial = dE_ddisp
                        if compute_virial:
                            data = replace_properties(data, virial=virial)
                        if compute_stress:
                            volume = torch.abs(torch.linalg.det(data.cell)).view(-1, 1, 1)
                            stress = virial / volume.clamp(min=1e-6)
                            data = replace_properties(data, stress=stress)

        return data
