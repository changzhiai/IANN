"""
UMA: Universal Models for Atoms (https://arxiv.org/abs/2506.23971).

Architecturally faithful single-file port of `fairchem.core.models.uma`
adapted to IANN's `AtomsData` interface and `Trainer` contract.

Architecture covers:
  - eSCN-MD backbone with dual SO3_Grid (lmax_lmax / lmax_mmax)
  - CSD conditioning (charge / spin / dataset embeddings + mix MLP)
  - Optional MoLE / MoE expert routing (eSCNMDMoeBackbone)
  - Output head zoo (MLP_Energy, Linear_Energy, Linear_Force, MLP_Stress)
  - IANN-style autograd forces/stress on edge_vectors / displacement

Charge / spin / dataset are taken from constructor kwargs (constants per
training run) to avoid touching IANN's AtomsData. The CSD/MoLE machinery
is fully wired and trainable; users with multi-task data can later extend
AtomsData and pass per-system tensors through.
"""
from __future__ import annotations

import copy
import functools
import math
import os
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.o3 import FromS2Grid, ToS2Grid

import iann
from iann.data import AtomsData, replace_properties

# ============================================================================
# Section 1: Radial utilities
# ============================================================================

@torch.jit.script
def gaussian(x: torch.Tensor, mean, std) -> torch.Tensor:
    a = (2 * math.pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class PolynomialEnvelope(torch.nn.Module):
    """Polynomial cutoff envelope (Behler-Parrinello style, p=5)."""

    def __init__(self, exponent: int = 5) -> None:
        super().__init__()
        assert exponent > 0
        self.p: float = float(exponent)
        self.a: float = -(self.p + 1) * (self.p + 2) / 2
        self.b: float = self.p * (self.p + 2)
        self.c: float = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        env_val = 1 + (d_scaled ** self.p) * (
            self.a + d_scaled * (self.b + self.c * d_scaled)
        )
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(env_val))


class GaussianSmearing(torch.nn.Module):
    """Gaussian radial basis."""

    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = (
            -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        )
        self.register_buffer("offset", offset, persistent=False)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class RadialMLP(nn.Module):
    """[Linear -> LayerNorm -> SiLU] stack ending in a Linear (no act)."""

    def __init__(self, channels_list) -> None:
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
            modules.append(torch.nn.SiLU())
        self.net = nn.Sequential(*modules)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


# ============================================================================
# Section 2: SO(3) utilities
# ============================================================================

class CoefficientMapping(torch.nn.Module):
    """Maps spherical harmonic indices to/from m-major ordering."""

    def __init__(self, lmax, mmax):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        l_harmonic = torch.tensor([]).long()
        m_harmonic = torch.tensor([]).long()
        m_complex = torch.tensor([]).long()

        for l in range(self.lmax + 1):
            mmax_l = min(self.mmax, l)
            m = torch.arange(-mmax_l, mmax_l + 1).long()
            m_complex = torch.cat([m_complex, m], dim=0)
            m_harmonic = torch.cat([m_harmonic, torch.abs(m).long()], dim=0)
            l_harmonic = torch.cat([l_harmonic, m.fill_(l).long()], dim=0)
        self.res_size = len(l_harmonic)

        num_coefficients = len(l_harmonic)
        to_m = torch.zeros([num_coefficients, num_coefficients])
        self.m_size = torch.zeros([self.mmax + 1]).long().tolist()

        offset = 0
        for m in range(self.mmax + 1):
            idx_r, idx_i = self.complex_idx(m, -1, m_complex, l_harmonic)
            for idx_out, idx_in in enumerate(idx_r):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_r)
            self.m_size[m] = len(idx_r)
            for idx_out, idx_in in enumerate(idx_i):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_i)

        self.register_buffer("l_harmonic", l_harmonic, persistent=False)
        self.register_buffer("m_harmonic", m_harmonic, persistent=False)
        self.register_buffer("m_complex", m_complex, persistent=False)
        self.register_buffer("to_m", to_m, persistent=False)
        self.pre_compute_coefficient_idx()

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
        lmax = self.lmax
        for l in range(lmax + 1):
            for m in range(lmax + 1):
                mask = torch.bitwise_and(self.l_harmonic.le(l), self.m_harmonic.le(m))
                indices = torch.arange(len(mask))
                mask_indices = torch.masked_select(indices, mask)
                self.register_buffer(
                    f"coefficient_idx_l{l}_m{m}", mask_indices, persistent=False
                )

    def prepare_coefficient_idx(self):
        lmax = self.lmax
        coefficient_idx_list = []
        for l in range(lmax + 1):
            l_list = []
            for m in range(lmax + 1):
                l_list.append(getattr(self, f"coefficient_idx_l{l}_m{m}", None))
            coefficient_idx_list.append(l_list)
        return coefficient_idx_list

    def coefficient_idx(self, lmax: int, mmax: int):
        if lmax > self.lmax or mmax > self.lmax:
            mask = torch.bitwise_and(
                self.l_harmonic.le(lmax), self.m_harmonic.le(mmax)
            )
            indices = torch.arange(len(mask), device=mask.device)
            return torch.masked_select(indices, mask)
        else:
            temp = self.prepare_coefficient_idx()
            return temp[lmax][mmax]


class SO3_Grid(torch.nn.Module):
    """Real spherical harmonic grid for to_grid / from_grid projection."""

    def __init__(self, lmax, mmax, normalization="integral", resolution=None, rescale=True):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.lat_resolution = 2 * (self.lmax + 1)
        if lmax == mmax:
            self.long_resolution = 2 * (self.mmax + 1) + 1
        else:
            self.long_resolution = 2 * (self.mmax) + 1
        if resolution is not None:
            self.lat_resolution = resolution
            self.long_resolution = resolution

        self.mapping = CoefficientMapping(self.lmax, self.lmax)
        self.rescale = rescale

        to_grid = ToS2Grid(
            self.lmax,
            (self.lat_resolution, self.long_resolution),
            normalization=normalization,
        )
        to_grid_mat = torch.einsum("mbi, am -> bai", to_grid.shb, to_grid.sha).detach()
        if rescale and lmax != mmax:
            for lval in range(lmax + 1):
                if lval <= mmax:
                    continue
                start_idx = lval ** 2
                length = 2 * lval + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                to_grid_mat[:, :, start_idx : (start_idx + length)] *= rescale_factor
        to_grid_mat = to_grid_mat[
            :, :, self.mapping.coefficient_idx(self.lmax, self.mmax)
        ]

        from_grid = FromS2Grid(
            (self.lat_resolution, self.long_resolution),
            self.lmax,
            normalization=normalization,
        )
        from_grid_mat = torch.einsum(
            "am, mbi -> bai", from_grid.sha, from_grid.shb
        ).detach()
        if rescale and lmax != mmax:
            for lval in range(lmax + 1):
                if lval <= mmax:
                    continue
                start_idx = lval ** 2
                length = 2 * lval + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                from_grid_mat[:, :, start_idx : (start_idx + length)] *= rescale_factor
        from_grid_mat = from_grid_mat[
            :, :, self.mapping.coefficient_idx(self.lmax, self.mmax)
        ]

        self.register_buffer("to_grid_mat", to_grid_mat, persistent=False)
        self.register_buffer("from_grid_mat", from_grid_mat, persistent=False)

    def to_grid(self, embedding, lmax, mmax):
        to_grid_mat = self.to_grid_mat[
            :, :, self.mapping.coefficient_idx(lmax, mmax)
        ]
        return torch.einsum("bai, zic -> zbac", to_grid_mat, embedding)

    def from_grid(self, grid, lmax, mmax):
        from_grid_mat = self.from_grid_mat[
            :, :, self.mapping.coefficient_idx(lmax, mmax)
        ]
        return torch.einsum("bai, zbac -> zic", from_grid_mat, grid)


# ============================================================================
# Section 3: Rotation utilities (Euler-path Wigner D)
# ============================================================================

class Safeacos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x_clamped = x.clamp(-1 + 1e-7, 1 - 1e-7)
        ctx.save_for_backward(x_clamped)
        return torch.acos(x_clamped)

    @staticmethod
    def backward(ctx, grad_output):
        (x_clamped,) = ctx.saved_tensors
        denom = torch.sqrt(1 - x_clamped.pow(2)).clamp(min=1e-7)
        return -grad_output / denom


class Safeatan2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, x):
        ctx.save_for_backward(y, x)
        return torch.atan2(y, x)

    @staticmethod
    def backward(ctx, grad_output):
        y, x = ctx.saved_tensors
        denom = (x.pow(2) + y.pow(2)).clamp(min=1e-7)
        return (x / denom) * grad_output, (-y / denom) * grad_output


def init_edge_rot_euler_angles(edge_distance_vec):
    """Compute Euler angles aligning the +z axis with the edge direction."""
    xyz = torch.nn.functional.normalize(edge_distance_vec).clamp(-1.0, 1.0)
    x, y, z = torch.split(xyz, 1, dim=1)
    beta = Safeacos.apply(y.squeeze(-1))
    alpha = Safeatan2.apply(x.squeeze(-1), z.squeeze(-1))
    gamma = torch.zeros_like(alpha)
    return -gamma, -beta, -alpha


def _z_rot_mat(angle: torch.Tensor, lv: int) -> torch.Tensor:
    M = angle.new_zeros((*angle.shape, 2 * lv + 1, 2 * lv + 1))
    inds = list(range(0, 2 * lv + 1))
    reversed_inds = list(range(2 * lv, -1, -1))
    frequencies = list(range(lv, -lv - 1, -1))
    for i in range(len(frequencies)):
        M[..., inds[i], reversed_inds[i]] = torch.sin(frequencies[i] * angle)
        M[..., inds[i], inds[i]] = torch.cos(frequencies[i] * angle)
    return M


def wigner_D(lv, alpha, beta, gamma, _Jd) -> torch.Tensor:
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[lv].to(dtype=alpha.dtype, device=alpha.device)
    Xa = _z_rot_mat(alpha, lv)
    Xb = _z_rot_mat(beta, lv)
    Xc = _z_rot_mat(gamma, lv)
    return Xa @ J @ Xb @ J @ Xc


def eulers_to_wigner(eulers, start_lmax, end_lmax, Jd) -> torch.Tensor:
    alpha, beta, gamma = eulers
    size = int((end_lmax + 1) ** 2) - int((start_lmax) ** 2)
    wigner = torch.zeros(
        len(alpha), size, size, device=alpha.device, dtype=alpha.dtype
    )
    start = 0
    for l in range(start_lmax, end_lmax + 1):
        block = wigner_D(l, alpha, beta, gamma, Jd)
        end = start + block.size()[1]
        wigner[:, start:end, start:end] = block
        start = end
    return wigner


# ============================================================================
# Section 4: Normalization
# ============================================================================

def get_l_to_all_m_expand_index(lmax: int):
    expand_index = torch.zeros([(lmax + 1) ** 2]).long()
    for lval in range(lmax + 1):
        start_idx = lval ** 2
        length = 2 * lval + 1
        expand_index[start_idx : (start_idx + length)] = lval
    return expand_index


class EquivariantLayerNormArray(nn.Module):
    """Per-degree LayerNorm: separate scale/bias for each L; centers L=0."""

    def __init__(self, lmax, num_channels, eps=1e-5, affine=True, normalization="component"):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(lmax + 1, num_channels))
            self.affine_bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)
        assert normalization in ["norm", "component"]
        self.normalization = normalization

    def forward(self, node_input):
        out = []
        for lval in range(self.lmax + 1):
            start_idx = lval ** 2
            length = 2 * lval + 1
            feature = node_input.narrow(1, start_idx, length)
            if lval == 0:
                feature_mean = torch.mean(feature, dim=2, keepdim=True)
                feature = feature - feature_mean
            if self.normalization == "norm":
                feature_norm = feature.pow(2).sum(dim=1, keepdim=True)
            elif self.normalization == "component":
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)
            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)
            feature_norm = (feature_norm + self.eps).pow(-0.5)
            if self.affine:
                weight = self.affine_weight.narrow(0, lval, 1).view(1, 1, -1)
                feature_norm = feature_norm * weight
            feature = feature * feature_norm
            if self.affine and lval == 0:
                feature = feature + self.affine_bias.view(1, 1, -1)
            out.append(feature)
        return torch.cat(out, dim=1)


class EquivariantLayerNormArraySphericalHarmonics(nn.Module):
    """LayerNorm for L=0; degree-balanced shared norm for L>0."""

    def __init__(
        self,
        lmax,
        num_channels,
        eps=1e-5,
        affine=True,
        normalization="component",
        std_balance_degrees=True,
    ):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.std_balance_degrees = std_balance_degrees
        self.norm_l0 = torch.nn.LayerNorm(num_channels, eps=eps, elementwise_affine=affine)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(lmax, num_channels))
        else:
            self.register_parameter("affine_weight", None)
        assert normalization in ["norm", "component"]
        self.normalization = normalization
        if std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2 - 1, 1)
            for lval in range(1, self.lmax + 1):
                start_idx = lval ** 2 - 1
                length = 2 * lval + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = 1.0 / length
            balance_degree_weight = balance_degree_weight / self.lmax
            self.register_buffer(
                "balance_degree_weight", balance_degree_weight, persistent=False
            )
        else:
            self.balance_degree_weight = None

    def forward(self, node_input):
        out = []
        feature = node_input.narrow(1, 0, 1)
        feature = self.norm_l0(feature)
        out.append(feature)
        if self.lmax > 0:
            num_m_components = (self.lmax + 1) ** 2
            feature = node_input.narrow(1, 1, num_m_components - 1)
            if self.normalization == "norm":
                feature_norm = feature.pow(2).sum(dim=1, keepdim=True)
            elif self.normalization == "component":
                if self.std_balance_degrees:
                    feature_norm = feature.pow(2)
                    feature_norm = torch.einsum(
                        "nic, ia -> nac", feature_norm, self.balance_degree_weight
                    )
                else:
                    feature_norm = feature.pow(2).mean(dim=1, keepdim=True)
            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)
            feature_norm = (feature_norm + self.eps).pow(-0.5)
            for lval in range(1, self.lmax + 1):
                start_idx = lval ** 2
                length = 2 * lval + 1
                feature = node_input.narrow(1, start_idx, length)
                if self.affine:
                    weight = self.affine_weight.narrow(0, lval - 1, 1).view(1, 1, -1)
                    feature_scale = feature_norm * weight
                else:
                    feature_scale = feature_norm
                feature = feature * feature_scale
                out.append(feature)
        return torch.cat(out, dim=1)


class EquivariantRMSNormArraySphericalHarmonics(nn.Module):
    """RMS norm across all m components for L>=0 (one scalar variance)."""

    def __init__(self, lmax, num_channels, eps=1e-5, affine=True, normalization="component"):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones((lmax + 1), num_channels))
        else:
            self.register_parameter("affine_weight", None)
        assert normalization in ["norm", "component"]
        self.normalization = normalization

    def forward(self, node_input):
        out = []
        feature = node_input
        if self.normalization == "norm":
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)
        elif self.normalization == "component":
            feature_norm = feature.pow(2).mean(dim=1, keepdim=True)
        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)
        feature_norm = (feature_norm + self.eps).pow(-0.5)
        for lval in range(self.lmax + 1):
            start_idx = lval ** 2
            length = 2 * lval + 1
            feature = node_input.narrow(1, start_idx, length)
            if self.affine:
                weight = self.affine_weight.narrow(0, lval, 1).view(1, 1, -1)
                feature_scale = feature_norm * weight
            else:
                feature_scale = feature_norm
            feature = feature * feature_scale
            out.append(feature)
        return torch.cat(out, dim=1)


class EquivariantRMSNormArraySphericalHarmonicsV2(nn.Module):
    """RMS norm across L>=0; expand weights so we can multiply in one shot."""

    def __init__(
        self,
        lmax,
        num_channels,
        eps=1e-5,
        affine=True,
        normalization="component",
        centering=True,
        std_balance_degrees=True,
    ):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.centering = centering
        self.std_balance_degrees = std_balance_degrees
        if affine:
            self.affine_weight = nn.Parameter(
                torch.ones((self.lmax + 1), self.num_channels)
            )
            self.affine_bias = (
                nn.Parameter(torch.zeros(self.num_channels)) if centering else None
            )
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)
        assert normalization in ["norm", "component"]
        self.normalization = normalization
        self.register_buffer(
            "expand_index", get_l_to_all_m_expand_index(self.lmax), persistent=False
        )
        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2, 1)
            for lval in range(self.lmax + 1):
                start_idx = lval ** 2
                length = 2 * lval + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = 1.0 / length
            balance_degree_weight /= self.lmax + 1
            self.register_buffer(
                "balance_degree_weight", balance_degree_weight, persistent=False
            )

    def forward(self, node_input):
        feature = node_input
        if self.centering:
            feature_l0 = feature.narrow(1, 0, 1)
            feature_l0_mean = feature_l0.mean(dim=2, keepdim=True)
            feature_l0 = feature_l0 - feature_l0_mean
            feature = torch.cat(
                (feature_l0, feature.narrow(1, 1, feature.shape[1] - 1)), dim=1
            )
        if self.normalization == "norm":
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)
        elif self.normalization == "component":
            if self.std_balance_degrees:
                feature_norm = torch.einsum(
                    "nic, ia -> nac", feature.pow(2), self.balance_degree_weight
                )
            else:
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)
        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)
        feature_norm = (feature_norm + self.eps).pow(-0.5)
        if self.affine:
            weight = torch.index_select(
                self.affine_weight.view(1, self.lmax + 1, self.num_channels),
                dim=1,
                index=self.expand_index,
            )
            feature_norm = feature_norm * weight
        out = feature * feature_norm
        if self.affine and self.centering:
            out[:, 0:1, :] = out.narrow(1, 0, 1) + self.affine_bias.view(
                1, 1, self.num_channels
            )
        return out


def get_normalization_layer(
    norm_type: str,
    lmax: int,
    num_channels: int,
    eps: float = 1e-5,
    affine: bool = True,
    normalization: str = "component",
):
    if norm_type == "layer_norm":
        return EquivariantLayerNormArray(lmax, num_channels, eps, affine, normalization)
    if norm_type == "layer_norm_sh":
        return EquivariantLayerNormArraySphericalHarmonics(
            lmax, num_channels, eps, affine, normalization
        )
    if norm_type == "rms_norm_sh":
        return EquivariantRMSNormArraySphericalHarmonicsV2(
            lmax, num_channels, eps, affine, normalization
        )
    if norm_type == "rms_norm_sh_v1":
        return EquivariantRMSNormArraySphericalHarmonics(
            lmax, num_channels, eps, affine, normalization
        )
    raise ValueError(f"Unknown norm_type: {norm_type}")


# ============================================================================
# Section 5: SO(3) Linear
# ============================================================================

class SO3_Linear(torch.nn.Module):
    """Per-degree linear layer; bias on L=0 only."""

    def __init__(self, in_features, out_features, lmax) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax
        self.weight = torch.nn.Parameter(
            torch.randn((self.lmax + 1), out_features, in_features)
        )
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        self.register_buffer(
            "expand_index", get_l_to_all_m_expand_index(lmax), persistent=False
        )

    def forward(self, input_embedding):
        weight = torch.index_select(self.weight, dim=0, index=self.expand_index)
        out = torch.einsum("bmi, moi -> bmo", input_embedding, weight).contiguous()
        out[:, 0:1, :] = out.narrow(1, 0, 1) + self.bias.view(1, 1, self.out_features)
        return out


# ============================================================================
# Section 6: SO(2) Convolution
# ============================================================================

class SO2_m_Conv(torch.nn.Module):
    """SO(2) conv on +/- m features for one m."""

    def __init__(self, m, sphere_channels, m_output_channels, lmax, mmax) -> None:
        super().__init__()
        self.m = m
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax
        assert mmax >= m
        num_coefficents = self.lmax - m + 1
        num_channels = num_coefficents * self.sphere_channels
        self.out_channels_half = self.m_output_channels * (
            num_channels // self.sphere_channels
        )
        self.fc = nn.Linear(num_channels, 2 * self.out_channels_half, bias=False)
        self.fc.weight.data.mul_(1 / math.sqrt(2))

    def forward(self, x_m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_m = self.fc(x_m)
        x_r_0, x_i_0, x_r_1, x_i_1 = x_m.reshape(
            x_m.shape[0], -1, self.out_channels_half
        ).split(1, dim=1)
        x_m_r = x_r_0 - x_i_1
        x_m_i = x_r_1 + x_i_0
        return (
            x_m_r.view(x_m.shape[0], -1, self.m_output_channels),
            x_m_i.view(x_m.shape[0], -1, self.m_output_channels),
        )


class SO2_Convolution(torch.nn.Module):
    """SO(2) block: per-m convs with optional radial gating."""

    def __init__(
        self,
        sphere_channels,
        m_output_channels,
        lmax,
        mmax,
        mappingReduced,
        internal_weights: bool = True,
        edge_channels_list=None,
        extra_m0_output_channels=None,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced
        self.internal_weights = internal_weights
        self.extra_m0_output_channels = extra_m0_output_channels
        self.edge_channels_list = (
            copy.deepcopy(edge_channels_list) if edge_channels_list is not None else None
        )

        num_channels_m0 = (lmax + 1) * sphere_channels
        m0_output_channels = m_output_channels * (lmax + 1)
        if extra_m0_output_channels is not None:
            m0_output_channels = m0_output_channels + extra_m0_output_channels
        self.fc_m0 = nn.Linear(num_channels_m0, m0_output_channels)
        num_channels_rad = self.fc_m0.in_features

        self.so2_m_conv = nn.ModuleList()
        for m in range(1, mmax + 1):
            self.so2_m_conv.append(
                SO2_m_Conv(m, sphere_channels, m_output_channels, lmax, mmax)
            )
            num_channels_rad += self.so2_m_conv[-1].fc.in_features

        self.rad_func = None
        if not self.internal_weights:
            assert edge_channels_list is not None
            ec = copy.deepcopy(edge_channels_list)
            ec.append(int(num_channels_rad))
            self.rad_func = RadialMLP(ec)

        self.m_split_sizes = [mappingReduced.m_size[0]] + (
            torch.tensor(mappingReduced.m_size[1:]) * 2
        ).tolist()
        self.edge_split_sizes = [self.fc_m0.in_features] + [
            mod.fc.in_features for mod in self.so2_m_conv
        ]

    def forward(self, x: torch.Tensor, x_edge: Optional[torch.Tensor] = None):
        if self.rad_func is not None:
            x_edge = self.rad_func(x_edge)

        x_by_m = x.split(self.m_split_sizes, dim=1)
        if x_edge is not None:
            x_edge_by_m = x_edge.split(self.edge_split_sizes, dim=1)

        num_edges = x.shape[0]
        x_0 = x_by_m[0].view(num_edges, -1)
        if x_edge is not None:
            x_0 = x_0 * x_edge_by_m[0]
        x_0 = self.fc_m0(x_0)

        if self.extra_m0_output_channels is not None:
            x_0_extra, x_0 = x_0.split(
                (
                    self.extra_m0_output_channels,
                    self.fc_m0.out_features - self.extra_m0_output_channels,
                ),
                -1,
            )

        out = [x_0.view(num_edges, -1, self.m_output_channels)]
        for m in range(1, self.mmax + 1):
            x_m = x_by_m[m].view(num_edges, 2, -1)
            if x_edge is not None:
                x_m = x_m * x_edge_by_m[m].unsqueeze(1)
            x_m_pair = self.so2_m_conv[m - 1](x_m)
            out.extend(x_m_pair)
        out = torch.cat(out, dim=1)

        if self.extra_m0_output_channels is not None:
            return out, x_0_extra
        return out


# ============================================================================
# Section 7: Activations
# ============================================================================

class ScaledSiLU(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.scale_factor = 1.6791767923989418

    def forward(self, inputs):
        return F.silu(inputs, inplace=self.inplace) * self.scale_factor


class GateActivation(torch.nn.Module):
    """Sigmoid-gated SiLU per-degree activation."""

    def __init__(self, lmax, mmax, num_channels, m_prime: bool = True) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.num_channels = num_channels
        num_components = 0
        for lval in range(1, lmax + 1):
            num_components += min((2 * lval + 1), (2 * mmax + 1))
        expand_index = torch.zeros([num_components]).long()
        if m_prime:
            start_idx = 0
            expand_index[0:lmax] = torch.arange(lmax)
            start_idx = lmax
            for mval in range(1, mmax + 1):
                length = 2 * (lmax + 1 - mval)
                expand_index[start_idx : (start_idx + length)] = torch.cat(
                    [torch.arange(mval - 1, lmax), torch.arange(mval - 1, lmax)]
                )
                start_idx += length
        else:
            start_idx = 0
            for lval in range(1, lmax + 1):
                length = min((2 * lval + 1), (2 * mmax + 1))
                expand_index[start_idx : (start_idx + length)] = lval - 1
                start_idx += length
        self.register_buffer("expand_index", expand_index, persistent=False)
        self.scalar_act = torch.nn.SiLU()
        self.gate_act = torch.nn.Sigmoid()

    def forward(self, gating_scalars, input_tensors):
        gating_scalars = self.gate_act(gating_scalars).view(
            gating_scalars.shape[0], self.lmax, self.num_channels
        )
        gating_scalars = torch.index_select(
            gating_scalars, dim=1, index=self.expand_index
        )
        scalars, vectors = input_tensors.split(
            (1, input_tensors.shape[1] - 1), 1
        )
        return torch.cat(
            (self.scalar_act(scalars), vectors * gating_scalars), dim=1
        )


class S2Activation_M(torch.nn.Module):
    """Pointwise SiLU on the S2 grid (m-projected)."""

    def __init__(self, lmax, mmax, SO3_grid, to_m) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.act = torch.nn.SiLU()
        self.SO3_grid = SO3_grid
        to_grid_mat = self.SO3_grid.to_grid_mat
        to_grid_mat_m = torch.einsum("ji,bai->jba", to_m, to_grid_mat)
        self.register_buffer("to_grid_mat_m", to_grid_mat_m, persistent=False)
        from_grid_mat = self.SO3_grid.from_grid_mat
        from_grid_mat_m = torch.einsum("ji,bai->baj", to_m, from_grid_mat)
        self.register_buffer("from_grid_mat_m", from_grid_mat_m, persistent=False)

    def forward(self, inputs):
        x_grid = torch.einsum("iba, zic -> zbac", self.to_grid_mat_m, inputs)
        x_grid = self.act(x_grid)
        return torch.einsum("bai, zbac -> zic", self.from_grid_mat_m, x_grid)


class SeparableS2Activation_M(torch.nn.Module):
    def __init__(self, lmax, mmax, SO3_grid, to_m) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.scalar_act = torch.nn.SiLU()
        self.s2_act = S2Activation_M(lmax, mmax, SO3_grid, to_m)

    def forward(self, input_scalars, input_tensors):
        output_scalars = self.scalar_act(input_scalars).reshape(
            input_scalars.shape[0], 1, -1
        )
        output_tensors = self.s2_act(input_tensors)
        return torch.cat(
            (output_scalars, output_tensors.narrow(1, 1, output_tensors.shape[1] - 1)),
            dim=1,
        )


# ============================================================================
# Section 8: Charge / Spin / Dataset embeddings
# ============================================================================

class ChgSpinEmbedding(nn.Module):
    """Embedding for integer charge or spin values.

    Three modes:
      - 'pos_emb': sinusoidal positional features (frozen unless grad=True).
      - 'lin_emb': single Linear from float input.
      - 'rand_emb': table lookup (offset + nn.Embedding).
    """

    def __init__(
        self,
        embedding_type: str,
        embedding_target: str,
        embedding_size: int,
        grad: bool,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        assert embedding_type in ["pos_emb", "lin_emb", "rand_emb"]
        self.embedding_type = embedding_type
        assert embedding_target in ["charge", "spin"]
        self.embedding_target = embedding_target
        assert embedding_size % 2 == 0, f"{embedding_size=} must be even"

        if embedding_target == "charge":
            self._index_offset = 100
            self._num_embeddings = 201
        else:
            self._index_offset = 0
            self._num_embeddings = 101

        if embedding_type == "pos_emb":
            self.W = nn.Parameter(
                torch.randn(embedding_size // 2) * scale, requires_grad=grad
            )
        elif embedding_type == "lin_emb":
            self.lin_emb = nn.Linear(in_features=1, out_features=embedding_size)
            if not grad:
                for p in self.lin_emb.parameters():
                    p.requires_grad = False
        elif embedding_type == "rand_emb":
            self.rand_emb = nn.Embedding(self._num_embeddings, embedding_size)
            if not grad:
                for p in self.rand_emb.parameters():
                    p.requires_grad = False

    def forward(self, x):
        if self.embedding_type == "pos_emb":
            x_proj = x[:, None].float() * self.W[None, :] * 2 * torch.pi
            if self.embedding_target == "charge":
                return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            zero_idxs = torch.where(x == 0)[0]
            emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            emb[zero_idxs] = 0
            return emb
        if self.embedding_type == "lin_emb":
            x = x.clone()
            if self.embedding_target == "spin":
                x[x == 0] = -100
            return self.lin_emb(x.unsqueeze(-1).float())
        # rand_emb
        indices = x + self._index_offset
        return self.rand_emb(indices.long())


class DatasetEmbedding(nn.Module):
    """Per-dataset embedding via a ModuleDict of nn.Embedding(1, C)."""

    def __init__(self, embedding_size: int, enable_grad: bool, dataset_mapping) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.enable_grad = enable_grad
        self.dataset_mapping = dataset_mapping
        self.dataset_emb_dict = nn.ModuleDict({})
        for dataset in dataset_mapping:
            if dataset not in self.dataset_emb_dict:
                self.dataset_emb_dict[dataset] = nn.Embedding(1, embedding_size)
                if not enable_grad:
                    for p in self.dataset_emb_dict[dataset].parameters():
                        p.requires_grad = False

    def forward(self, dataset_list):
        device = list(self.parameters())[0].device
        emb_idx = torch.tensor(0, device=device, dtype=torch.long)
        dataset_list = [self.dataset_mapping[d] for d in dataset_list]
        if self.enable_grad and self.training:
            safety_loss_emb = torch.stack(
                [
                    self.dataset_emb_dict[d](emb_idx) * 0.0
                    for d in self.dataset_emb_dict
                ]
            ).sum(dim=0)
            emb_for_datasets = [
                (
                    self.dataset_emb_dict[d](emb_idx) + safety_loss_emb
                    if i == 0
                    else self.dataset_emb_dict[d](emb_idx)
                )
                for i, d in enumerate(dataset_list)
            ]
        else:
            emb_for_datasets = [
                self.dataset_emb_dict[d](emb_idx) for d in dataset_list
            ]
        return torch.stack(emb_for_datasets, dim=0)


# ============================================================================
# Section 9: Edge degree embedding
# ============================================================================

class EdgeDegreeEmbedding(torch.nn.Module):
    """Edge-feature scatter into m=0 SO(3) coefficients."""

    def __init__(
        self,
        sphere_channels: int,
        lmax: int,
        mmax: int,
        edge_channels_list,
        rescale_factor: float,
        mappingReduced,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced
        self.m_0_num_coefficients: int = self.mappingReduced.m_size[0]
        self.m_all_num_coefficents: int = len(self.mappingReduced.l_harmonic)
        ec = copy.deepcopy(edge_channels_list)
        ec.append(self.m_0_num_coefficients * self.sphere_channels)
        self.rad_func = RadialMLP(ec)
        self.rescale_factor = rescale_factor

    def forward(self, x, x_edge, edge_index, wigner_inv_envelope):
        radial = self.rad_func(x_edge).reshape(
            -1, self.m_0_num_coefficients, self.sphere_channels
        )
        wigner_inv_m0 = wigner_inv_envelope[:, :, : self.m_0_num_coefficients]
        x_edge_embedding = torch.bmm(wigner_inv_m0, radial)
        return x.index_add(
            0, edge_index[1], x_edge_embedding / self.rescale_factor
        )


# ============================================================================
# Section 10: Backend ops (general)
# ============================================================================

def prepare_wigner(wigner, wigner_inv, mappingReduced, coefficient_index=None):
    if coefficient_index is not None:
        wigner = wigner.index_select(1, coefficient_index)
        wigner_inv = wigner_inv.index_select(2, coefficient_index)
    wigner = torch.einsum("mk,nkj->nmj", mappingReduced.to_m.to(wigner.dtype), wigner)
    wigner_inv = torch.einsum(
        "njk,mk->njm", wigner_inv, mappingReduced.to_m.to(wigner_inv.dtype)
    )
    return wigner, wigner_inv


def node_to_edge_wigner_permute(x_full, edge_index, wigner):
    x_source = x_full[edge_index[0]]
    x_target = x_full[edge_index[1]]
    x_message = torch.cat((x_source, x_target), dim=2)
    return torch.bmm(wigner, x_message)


def permute_wigner_inv_edge_to_node(x_message, wigner_inv, edge_index, num_nodes):
    x_rotated = torch.bmm(wigner_inv, x_message)
    new_embedding = torch.zeros(
        (num_nodes,) + x_rotated.shape[1:],
        dtype=x_rotated.dtype,
        device=x_rotated.device,
    )
    new_embedding.index_add_(0, edge_index[1], x_rotated)
    return new_embedding


# ============================================================================
# Section 11: Channel balancing
# ============================================================================

def validate_contiguous_channels(channels, name: str):
    if not channels:
        return 0, 0
    sorted_channels = sorted(channels)
    expected = list(range(sorted_channels[0], sorted_channels[-1] + 1))
    if sorted_channels != expected:
        raise ValueError(f"{name} must be contiguous, got {channels}")
    return sorted_channels[0], sorted_channels[-1] + 1


def balance_channels_batched(
    emb: torch.Tensor,
    target: torch.Tensor,
    natoms: torch.Tensor,
    batch: torch.Tensor,
    start_idx: int,
    end_idx: int,
    target_offset: float = 0.0,
) -> torch.Tensor:
    """Constrain a contiguous channel range to sum to a target per system."""
    out_emb = emb.clone()
    num_systems = len(natoms)
    n_channels = end_idx - start_idx
    channels_to_balance = emb[:, 0, start_idx:end_idx]
    system_sums = torch.zeros(
        num_systems, n_channels, device=emb.device, dtype=emb.dtype
    )
    system_sums.index_add_(0, batch, channels_to_balance)
    target_sums = (target - target_offset).unsqueeze(1).expand(-1, n_channels)
    corrections = (system_sums - target_sums) / natoms.unsqueeze(1)
    out_emb[:, 0, start_idx:end_idx] = channels_to_balance - corrections[batch]
    return out_emb


# ============================================================================
# Section 12: MoLE (Mixture of Linear Experts) layer
# ============================================================================

def _interval_intersection(interval1, interval2):
    a, b = interval1
    c, d = interval2
    start = max(a, c)
    end = min(b, d)
    if start <= end:
        return [start, end]
    return None


def _mole_softmax(x):
    return torch.softmax(x, dim=1) + 0.005


def _mole_pnorm(x):
    return torch.nn.functional.normalize(x.abs() + 2 / x.shape[0], p=1.0, dim=1)


def norm_str_to_fn(act):
    if act == "softmax":
        return _mole_softmax
    if act == "pnorm":
        return _mole_pnorm
    raise ValueError(f"Unknown norm fn: {act}")


@dataclass
class MOLEGlobals:
    """Per-forward-call globals that all MOLE layers share."""

    expert_mixing_coefficients: Optional[torch.Tensor] = None
    mole_sizes: Optional[torch.Tensor] = None
    ac_start_idx: int = 0


def _init_mole_linear(num_experts, use_bias, out_features, in_features):
    k = math.sqrt(1.0 / in_features)
    weights = nn.Parameter(
        k * 2 * (torch.rand(num_experts, out_features, in_features) - 0.5)
    )
    bias = nn.Parameter(k * 2 * (torch.rand(out_features) - 0.5)) if use_bias else None
    return weights, bias


class MOLE(torch.nn.Module):
    """Mixture-of-experts Linear: per-system expert weights mixed at runtime."""

    def __init__(
        self,
        num_experts,
        in_features,
        out_features,
        global_mole_tensors: MOLEGlobals,
        bias: bool,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weights, self.bias = _init_mole_linear(
            num_experts, bias, out_features, in_features
        )
        self.global_mole_tensors = global_mole_tensors

    def merged_linear_layer(self):
        linear = torch.nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias is not None,
        ).to(self.weights.device)
        with torch.autocast(device_type=self.weights.device.type, enabled=False):
            weights = torch.einsum(
                "eoi, be->boi",
                self.weights,
                self.global_mole_tensors.expert_mixing_coefficients,
            )
        with torch.no_grad():
            linear.weight.copy_(weights[0])
            if self.bias is not None:
                linear.bias.copy_(self.bias)
        return linear

    def forward(self, x):
        with torch.autocast(device_type=self.weights.device.type, enabled=False):
            weights = torch.einsum(
                "eoi, be->boi",
                self.weights,
                self.global_mole_tensors.expert_mixing_coefficients,
            )

        out = []
        ac_start_idx = self.global_mole_tensors.ac_start_idx
        assert self.global_mole_tensors.mole_sizes is not None
        assert len(self.global_mole_tensors.mole_sizes) > 0
        start_idxs = [0] + torch.cumsum(
            self.global_mole_tensors.mole_sizes, dim=0
        ).tolist()
        mole_intervals = list(zip(start_idxs, start_idxs[1:]))

        input_segment = (ac_start_idx, ac_start_idx + x.shape[0])

        for n, mole_segment in enumerate(mole_intervals):
            interval_overlap = _interval_intersection(input_segment, mole_segment)
            if interval_overlap is not None:
                start = interval_overlap[0] - ac_start_idx
                end = interval_overlap[1] - ac_start_idx
                out.append(F.linear(x[start:end], weights[n], bias=self.bias))

        result = torch.cat(out, dim=0)
        assert result.shape[0] == x.shape[0]
        return result


# ============================================================================
# Section 13: MoLE model surgery
# ============================================================================

class MOLEInterface:
    """Mixin tagging a backbone as MoLE-aware."""

    def set_MOLE_coefficients(self, atomic_numbers_full, batch_full, csd_mixed_emb) -> None:
        return None

    def set_MOLE_sizes(self, nsystems, batch_full, edge_index) -> None:
        return None

    def log_MOLE_stats(self) -> None:
        return None

    def merge_MOLE_model(self, data):
        return self


def recursive_replace_so2m0_linear(model, replacement_factory):
    for _, child in model.named_children():
        if isinstance(child, torch.nn.Module):
            recursive_replace_so2m0_linear(child, replacement_factory)
        if isinstance(child, SO2_Convolution):
            target_device = child.fc_m0.weight.device
            child.fc_m0 = replacement_factory(child.fc_m0).to(target_device)


def recursive_replace_so2_MOLE(model, replacement_factory):
    for _, child in model.named_children():
        if isinstance(child, torch.nn.Module):
            recursive_replace_so2_MOLE(child, replacement_factory)
        if isinstance(child, SO2_Convolution):
            target_device = child.fc_m0.weights.device
            child.fc_m0 = replacement_factory(child.fc_m0).to(target_device)
            for so2_module in child.so2_m_conv:
                so2_module.fc = replacement_factory(so2_module.fc).to(target_device)


def recursive_replace_so2_linear(model, replacement_factory):
    for _, child in model.named_children():
        if isinstance(child, torch.nn.Module):
            recursive_replace_so2_linear(child, replacement_factory)
        if isinstance(child, SO2_Convolution):
            target_device = child.fc_m0.weight.device
            child.fc_m0 = replacement_factory(child.fc_m0).to(target_device)
            for so2_module in child.so2_m_conv:
                so2_module.fc = replacement_factory(so2_module.fc).to(target_device)


def recursive_replace_all_linear(model, replacement_factory):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.Linear):
            target_device = child.weight.device
            setattr(model, child_name, replacement_factory(child).to(target_device))
        elif isinstance(child, torch.nn.Module):
            recursive_replace_all_linear(child, replacement_factory)


def recursive_replace_notso2_linear(model, replacement_factory):
    for child_name, child in model.named_children():
        if isinstance(child, SO2_Convolution):
            continue
        if isinstance(child, torch.nn.Linear):
            target_device = child.weight.device
            setattr(model, child_name, replacement_factory(child).to(target_device))
        elif isinstance(child, torch.nn.Module):
            recursive_replace_notso2_linear(child, replacement_factory)


def model_search_and_replace(model, module_search_function, replacement_factory, layers=None):
    if layers is None:
        layers = list(range(len(model.blocks)))
    for layer_idx in layers:
        module_search_function(model.blocks[layer_idx], replacement_factory)


def replace_MOLE_with_linear(existing_mole_module: MOLE):
    return existing_mole_module.merged_linear_layer()


def replace_linear_with_MOLE(
    existing_linear_module,
    global_mole_tensors,
    num_experts,
    mole_layer_type,
    cache=None,
):
    layer_identifier = (
        existing_linear_module.in_features,
        existing_linear_module.out_features,
        existing_linear_module.bias,
    )
    if cache is not None and layer_identifier in cache:
        return cache[layer_identifier]
    if mole_layer_type != "pytorch":
        raise ValueError("Only mole_layer_type='pytorch' is supported in this port")
    layer = MOLE(
        num_experts=num_experts,
        global_mole_tensors=global_mole_tensors,
        in_features=existing_linear_module.in_features,
        out_features=existing_linear_module.out_features,
        bias=existing_linear_module.bias is not None,
    )
    if cache is not None:
        cache[layer_identifier] = layer
    return layer


def convert_model_to_MOLE_model(
    model,
    num_experts: int = 8,
    mole_dropout: float = 0.0,
    mole_expert_coefficient_norm: str = "softmax",
    act=torch.nn.SiLU,
    layers_mole=None,
    use_composition_embedding: bool = False,
    composition_dropout: float = 0.0,
    mole_layer_type: str = "pytorch",
    mole_single: bool = False,
    mole_type: str = "so2",
):
    """Replace target Linear layers in `model.blocks` with MOLE layers."""
    model.num_experts = num_experts
    if model.num_experts == 0:
        return
    model.mole_type = mole_type

    routing_mlp_dim = (use_composition_embedding + 1) * model.sphere_channels
    model.routing_mlp = nn.Sequential(
        nn.Linear(routing_mlp_dim, num_experts * 2, bias=True),
        nn.SiLU(),
        nn.Linear(num_experts * 2, num_experts * 2, bias=True),
        nn.SiLU(),
        nn.Linear(num_experts * 2, num_experts, bias=True),
        nn.SiLU(),
    )
    model.use_composition_embedding = use_composition_embedding
    model.composition_dropout = composition_dropout
    model.global_mole_tensors = MOLEGlobals(
        expert_mixing_coefficients=None, mole_sizes=None
    )
    model.mole_dropout = torch.nn.Dropout(mole_dropout)
    model.mole_expert_coefficient_norm = norm_str_to_fn(mole_expert_coefficient_norm)
    model.act = act()
    if model.use_composition_embedding:
        model.composition_embedding = nn.Embedding(
            model.max_num_elements, model.sphere_channels
        )
    model.counter = 0

    replacement_factory = functools.partial(
        replace_linear_with_MOLE,
        num_experts=model.num_experts,
        global_mole_tensors=model.global_mole_tensors,
        mole_layer_type=mole_layer_type,
        cache={} if mole_single else None,
    )

    if mole_type == "so2":
        model_search_and_replace(model, recursive_replace_so2_linear, replacement_factory, layers=layers_mole)
    elif mole_type == "so2m0":
        model_search_and_replace(model, recursive_replace_so2m0_linear, replacement_factory, layers=layers_mole)
    elif mole_type == "all":
        model_search_and_replace(model, recursive_replace_all_linear, replacement_factory, layers=layers_mole)
    elif mole_type == "notso2":
        model_search_and_replace(model, recursive_replace_notso2_linear, replacement_factory, layers=layers_mole)
    else:
        raise ValueError(f"Not a valid mole_type {mole_type}")


# ============================================================================
# Section 14: Blocks (Edgewise / SpectralAtomwise / GridAtomwise / eSCNMD_Block)
# ============================================================================

class Edgewise(nn.Module):
    """Per-edge SO(2) message passing (rotated source+target features)."""

    def __init__(
        self,
        sphere_channels,
        hidden_channels,
        lmax,
        mmax,
        edge_channels_list,
        mappingReduced,
        SO3_grid,
        act_type: str = "gate",
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced
        self.SO3_grid = SO3_grid
        self.act_type = act_type

        if act_type == "gate":
            self.act = GateActivation(lmax=lmax, mmax=mmax, num_channels=hidden_channels, m_prime=True)
            extra_m0_output_channels = lmax * hidden_channels
        elif act_type == "s2":
            self.act = SeparableS2Activation_M(
                lmax=lmax, mmax=mmax, SO3_grid=SO3_grid, to_m=mappingReduced.to_m
            )
            extra_m0_output_channels = hidden_channels
        else:
            raise ValueError(f"Unknown act_type {act_type}")

        self.so2_conv_1 = SO2_Convolution(
            2 * sphere_channels,
            hidden_channels,
            lmax,
            mmax,
            mappingReduced,
            internal_weights=False,
            edge_channels_list=copy.deepcopy(edge_channels_list),
            extra_m0_output_channels=extra_m0_output_channels,
        )
        self.so2_conv_2 = SO2_Convolution(
            hidden_channels,
            sphere_channels,
            lmax,
            mmax,
            mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=None,
        )

    def forward(self, x, x_edge, edge_index, wigner, wigner_inv_envelope):
        x_full = x  # graph-parallel disabled in this port
        x_message = node_to_edge_wigner_permute(x_full, edge_index, wigner)
        x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)
        x_message = self.act(x_0_gating, x_message)
        x_message = self.so2_conv_2(x_message)
        return permute_wigner_inv_edge_to_node(
            x_message, wigner_inv_envelope, edge_index, x.shape[0]
        )


class SpectralAtomwise(nn.Module):
    """Per-atom feed-forward in spherical-harmonic spectral space."""

    def __init__(self, sphere_channels, hidden_channels, lmax, mmax, SO3_grid):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.SO3_grid = SO3_grid
        self.scalar_mlp = nn.Sequential(
            nn.Linear(sphere_channels, lmax * hidden_channels, bias=True),
            nn.SiLU(),
        )
        self.so3_linear_1 = SO3_Linear(sphere_channels, hidden_channels, lmax=lmax)
        self.act = GateActivation(lmax=lmax, mmax=lmax, num_channels=hidden_channels)
        self.so3_linear_2 = SO3_Linear(hidden_channels, sphere_channels, lmax=lmax)

    def forward(self, x):
        gating_scalars = self.scalar_mlp(x.narrow(1, 0, 1))
        x = self.so3_linear_1(x)
        x = self.act(gating_scalars, x)
        return self.so3_linear_2(x)


class GridAtomwise(nn.Module):
    """Per-atom feed-forward on the S2 grid: to_grid -> MLP -> from_grid."""

    def __init__(self, sphere_channels, hidden_channels, lmax, mmax, SO3_grid):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.SO3_grid = SO3_grid
        self.grid_mlp = nn.Sequential(
            nn.Linear(sphere_channels, hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_channels, sphere_channels, bias=False),
        )

    def forward(self, x):
        x_grid = self.SO3_grid["lmax_lmax"].to_grid(x, self.lmax, self.lmax)
        x_grid = self.grid_mlp(x_grid)
        return self.SO3_grid["lmax_lmax"].from_grid(x_grid, self.lmax, self.lmax)


class eSCNMD_Block(nn.Module):
    """One block: (norm + edgewise + residual) -> (norm + atomwise + residual)."""

    def __init__(
        self,
        sphere_channels,
        hidden_channels,
        lmax,
        mmax,
        mappingReduced,
        SO3_grid,
        edge_channels_list,
        norm_type: str = "rms_norm_sh",
        act_type: str = "gate",
        ff_type: str = "grid",
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax

        self.norm_1 = get_normalization_layer(norm_type, lmax=lmax, num_channels=sphere_channels)
        # use lmax_mmax grid for edge SO(2) (only relevant when act_type='s2')
        edge_grid = SO3_grid["lmax_mmax"] if isinstance(SO3_grid, nn.ModuleDict) else SO3_grid
        self.edge_wise = Edgewise(
            sphere_channels=sphere_channels,
            hidden_channels=hidden_channels,
            lmax=lmax,
            mmax=mmax,
            edge_channels_list=edge_channels_list,
            mappingReduced=mappingReduced,
            SO3_grid=edge_grid,
            act_type=act_type,
        )
        self.norm_2 = get_normalization_layer(norm_type, lmax=lmax, num_channels=sphere_channels)

        if ff_type == "spectral":
            self.atom_wise = SpectralAtomwise(
                sphere_channels=sphere_channels,
                hidden_channels=hidden_channels,
                lmax=lmax,
                mmax=mmax,
                SO3_grid=SO3_grid,
            )
        elif ff_type == "grid":
            self.atom_wise = GridAtomwise(
                sphere_channels=sphere_channels,
                hidden_channels=hidden_channels,
                lmax=lmax,
                mmax=mmax,
                SO3_grid=SO3_grid,
            )
        else:
            raise ValueError(f"Unknown ff_type {ff_type}")

    def forward(
        self,
        x,
        x_edge,
        edge_index,
        wigner,
        wigner_inv_envelope,
        sys_node_embedding: Optional[torch.Tensor] = None,
    ):
        x_res = x
        x = self.norm_1(x)
        if sys_node_embedding is not None:
            x = x.clone()
            x[:, 0, :] = x[:, 0, :] + sys_node_embedding
        x = self.edge_wise(x, x_edge, edge_index, wigner, wigner_inv_envelope)
        x = x + x_res

        x_res = x
        x = self.norm_2(x)
        x = self.atom_wise(x)
        x = x + x_res
        return x


# ============================================================================
# Section 15: Output helpers (energy reduction, irreps, stress recomposition)
# ============================================================================

def get_l_component_range(x: torch.Tensor, l_min: int, l_max: int) -> torch.Tensor:
    """Slice spherical harmonic components for L in [l_min, l_max]."""
    start_idx = l_min * l_min
    num_components = (l_max + 1) ** 2 - l_min ** 2
    return x.narrow(1, start_idx, num_components)


def reduce_node_to_system(
    node_values: torch.Tensor,
    batch: torch.Tensor,
    num_systems: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sum node values into per-system values."""
    output_shape = (num_systems,) + node_values.shape[1:]
    system_values = torch.zeros(
        output_shape, device=node_values.device, dtype=torch.float64
    )
    if node_values.dim() == 1:
        system_values.index_add_(0, batch, node_values.to(system_values.dtype))
    else:
        flat_node = node_values.view(node_values.shape[0], -1)
        flat_system = system_values.view(num_systems, -1)
        flat_system.index_add_(0, batch, flat_node.to(flat_system.dtype))
        system_values = flat_system.view(output_shape)
    return system_values, system_values


def compute_energy(
    emb: dict,
    energy_block: torch.nn.Module,
    batch: torch.Tensor,
    num_systems: int,
    natoms: Optional[torch.Tensor] = None,
    reduce: str = "sum",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-atom energy_block(L=0) -> scatter-sum to system level."""
    scalar_embedding = get_l_component_range(
        emb["node_embedding"], l_min=0, l_max=0
    ).squeeze(1)
    node_energy = energy_block(scalar_embedding)
    node_energy_flat = node_energy.view(-1)
    energy, energy_part = reduce_node_to_system(
        node_energy_flat, batch, num_systems
    )
    if reduce == "sum":
        pass
    elif reduce == "mean":
        if natoms is None:
            raise ValueError("natoms must be provided when reduce='mean'")
        energy = energy / natoms
    else:
        raise ValueError(f"reduce must be sum or mean, got {reduce}")
    return energy, energy_part


def irreps_sum(L_max: int) -> int:
    """Number of spherical harmonic coefficients up to and including L_max."""
    total = 0
    for L in range(L_max + 1):
        total += 2 * L + 1
    return total


def cg_change_mat(ang_mom: int, device: str = "cpu") -> torch.Tensor:
    """Change of basis from rank-2 tensor to spherical harmonic L=0 + L=2."""
    if ang_mom != 2:
        raise NotImplementedError("Only L=2 is implemented (used by stress head)")
    change_mat = torch.tensor(
        [
            [3 ** (-0.5), 0, 0, 0, 3 ** (-0.5), 0, 0, 0, 3 ** (-0.5)],
            [0, 0, 0, 0, 0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0],
            [0, 0, -(2 ** (-0.5)), 0, 0, 0, 2 ** (-0.5), 0, 0],
            [0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0, 0, 0, 0, 0],
            [0, 0, 0.5 ** 0.5, 0, 0, 0, 0.5 ** 0.5, 0, 0],
            [0, 0.5 ** 0.5, 0, 0.5 ** 0.5, 0, 0, 0, 0, 0],
            [
                -(6 ** (-0.5)),
                0,
                0,
                0,
                2 * 6 ** (-0.5),
                0,
                0,
                0,
                -(6 ** (-0.5)),
            ],
            [0, 0, 0, 0, 0, 0.5 ** 0.5, 0, 0.5 ** 0.5, 0],
            [-(2 ** (-0.5)), 0, 0, 0, 0, 0, 0, 0, 2 ** (-0.5)],
        ],
        device=device,
    ).detach()
    return change_mat


def compose_tensor(trace: torch.Tensor, l2_symmetric: torch.Tensor) -> torch.Tensor:
    """Recompose a rank-2 stress tensor from its L=0 trace and L=2 symmetric part."""
    if trace.shape[1] != 1:
        raise ValueError("trace must be shape (B, 1)")
    if l2_symmetric.shape[1] != 5:
        raise ValueError("l2_symmetric must be shape (B, 5)")
    if trace.shape[0] != l2_symmetric.shape[0]:
        raise ValueError("Batch shape mismatch")
    batch_size = trace.shape[0]
    decomposed_preds = torch.zeros(
        batch_size, irreps_sum(2), device=trace.device, dtype=trace.dtype
    )
    decomposed_preds[:, : irreps_sum(0)] = trace
    decomposed_preds[:, irreps_sum(1) : irreps_sum(2)] = l2_symmetric
    r2_tensor = torch.einsum(
        "ba, cb->ca",
        cg_change_mat(2, device=str(trace.device)),
        decomposed_preds,
    )
    return r2_tensor


# ============================================================================
# Section 16: Output heads
# ============================================================================

class MLP_Energy_Head(nn.Module):
    """3-layer MLP energy head; per-atom L=0 embedding -> 1 scalar -> scatter-sum."""

    def __init__(self, sphere_channels, hidden_channels, reduce: str = "sum") -> None:
        super().__init__()
        self.reduce = reduce
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.energy_block = nn.Sequential(
            nn.Linear(sphere_channels, hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1, bias=True),
        )

    def forward(self, emb: dict, num_systems: int, natoms: torch.Tensor) -> torch.Tensor:
        energy, _ = compute_energy(
            emb,
            self.energy_block,
            emb["batch"],
            num_systems,
            natoms=natoms,
            reduce=self.reduce,
        )
        return energy


class Linear_Energy_Head(nn.Module):
    """Single Linear -> 1-scalar energy head."""

    def __init__(self, sphere_channels: int, reduce: str = "sum") -> None:
        super().__init__()
        self.reduce = reduce
        self.energy_block = nn.Linear(sphere_channels, 1, bias=True)

    def forward(self, emb: dict, num_systems: int, natoms: torch.Tensor) -> torch.Tensor:
        energy, _ = compute_energy(
            emb,
            self.energy_block,
            emb["batch"],
            num_systems,
            natoms=natoms,
            reduce=self.reduce,
        )
        return energy


class Linear_Force_Head(nn.Module):
    """Direct (non-autograd) force head: SO3_Linear on L=0+L=1 -> L=1 vector."""

    def __init__(self, sphere_channels: int) -> None:
        super().__init__()
        self.linear = SO3_Linear(sphere_channels, 1, lmax=1)

    def forward(self, emb: dict) -> torch.Tensor:
        l0_l1 = get_l_component_range(emb["node_embedding"], l_min=0, l_max=1)
        forces_output = self.linear(l0_l1)
        forces = get_l_component_range(forces_output, l_min=1, l_max=1)
        return forces.view(-1, 3).contiguous()


class MLP_Stress_Head(nn.Module):
    """Direct stress head: predicts L=0 (trace) and L=2 (symmetric) separately."""

    def __init__(self, sphere_channels: int, hidden_channels: int, reduce: str = "mean") -> None:
        super().__init__()
        assert reduce in ["sum", "mean"]
        self.reduce = reduce
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.scalar_block = nn.Sequential(
            nn.Linear(sphere_channels, hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1, bias=True),
        )
        self.l2_linear = SO3_Linear(sphere_channels, 1, lmax=2)

    def forward(self, emb: dict, num_systems: int, natoms: torch.Tensor) -> torch.Tensor:
        batch = emb["batch"]

        scalar_embedding = get_l_component_range(
            emb["node_embedding"], l_min=0, l_max=0
        ).squeeze(1)
        node_scalar = self.scalar_block(scalar_embedding).view(-1)
        iso_stress, _ = reduce_node_to_system(node_scalar, batch, num_systems)
        if self.reduce == "mean":
            iso_stress = iso_stress / natoms

        l0l1l2_embedding = get_l_component_range(
            emb["node_embedding"], l_min=0, l_max=2
        )
        l2_output = self.l2_linear(l0l1l2_embedding)
        node_l2 = (
            get_l_component_range(l2_output, l_min=2, l_max=2).view(-1, 5).contiguous()
        )
        aniso_stress, _ = reduce_node_to_system(node_l2, batch, num_systems)
        if self.reduce == "mean":
            aniso_stress = aniso_stress / natoms.unsqueeze(1)

        stress = compose_tensor(iso_stress.unsqueeze(1).float(), aniso_stress.float())
        return stress


# ============================================================================
# Section 17: Backbone (eSCNMDBackbone)
# ============================================================================

class eSCNMDBackbone(nn.Module):
    """eSCN-MD backbone with CSD conditioning and dual SO3 grids.

    Forward consumes a dict ("data_dict") and returns
    {"node_embedding", "batch"}. The IANN adapter (UMA class) builds
    that data_dict from AtomsData and applies output heads.
    """

    def __init__(
        self,
        max_num_elements: int = 100,
        sphere_channels: int = 128,
        lmax: int = 2,
        mmax: int = 2,
        grid_resolution: Optional[int] = None,
        cutoff: float = 5.0,
        edge_channels: int = 128,
        num_distance_basis: int = 512,
        num_layers: int = 2,
        hidden_channels: int = 128,
        norm_type: str = "rms_norm_sh",
        act_type: str = "gate",
        ff_type: str = "grid",
        chg_spin_emb_type: str = "pos_emb",
        cs_emb_grad: bool = False,
        dataset_emb_grad: bool = False,
        dataset_mapping: Optional[dict] = None,
        use_dataset_embedding: bool = True,
        charge_balanced_channels: Optional[List[int]] = None,
        spin_balanced_channels: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.max_num_elements = max_num_elements
        self.lmax = lmax
        self.mmax = mmax
        self.sphere_channels = sphere_channels
        self.grid_resolution = grid_resolution
        self.cutoff = cutoff

        # MoLE-related attrs (set by convert_model_to_MOLE_model when applicable)
        self.num_experts = 0
        self.global_mole_tensors: Optional[MOLEGlobals] = None

        # Channel balancing
        cc = list(charge_balanced_channels) if charge_balanced_channels else []
        sc = list(spin_balanced_channels) if spin_balanced_channels else []
        self.charge_channel_start, self.charge_channel_end = (
            validate_contiguous_channels(cc, "charge_balanced_channels")
        )
        self.spin_channel_start, self.spin_channel_end = (
            validate_contiguous_channels(sc, "spin_balanced_channels")
        )

        # Wigner Jd buffers
        Jd_path = os.path.join(iann.__path__[0], "data", "Jd.pt")
        Jd_list = torch.load(Jd_path, weights_only=True)
        for l in range(self.lmax + 1):
            self.register_buffer(f"Jd_{l}", Jd_list[l])

        self.sph_feature_size = int((self.lmax + 1) ** 2)
        self.mappingReduced = CoefficientMapping(self.lmax, self.mmax)

        self.SO3_grid = nn.ModuleDict()
        self.SO3_grid["lmax_lmax"] = SO3_Grid(
            self.lmax, self.lmax, resolution=grid_resolution, rescale=True
        )
        self.SO3_grid["lmax_mmax"] = SO3_Grid(
            self.lmax, self.mmax, resolution=grid_resolution, rescale=True
        )

        self.sphere_embedding = nn.Embedding(self.max_num_elements, self.sphere_channels)

        self.use_dataset_embedding = use_dataset_embedding
        self.dataset_mapping = dataset_mapping or {"oc20": "oc20"}

        self.charge_embedding = ChgSpinEmbedding(
            chg_spin_emb_type, "charge", self.sphere_channels, grad=cs_emb_grad
        )
        self.spin_embedding = ChgSpinEmbedding(
            chg_spin_emb_type, "spin", self.sphere_channels, grad=cs_emb_grad
        )

        if self.use_dataset_embedding:
            self.dataset_embedding = DatasetEmbedding(
                self.sphere_channels,
                enable_grad=dataset_emb_grad,
                dataset_mapping=self.dataset_mapping,
            )
            self.mix_csd = nn.Linear(3 * self.sphere_channels, self.sphere_channels)
        else:
            self.mix_csd = nn.Linear(2 * self.sphere_channels, self.sphere_channels)

        self.cutoff = cutoff
        self.edge_channels = edge_channels
        self.num_distance_basis = num_distance_basis
        self.distance_expansion = GaussianSmearing(
            0.0, self.cutoff, self.num_distance_basis, 2.0
        )

        self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        self.edge_channels_list = [
            self.num_distance_basis + 2 * self.edge_channels,
            self.edge_channels,
            self.edge_channels,
        ]

        self.edge_degree_embedding = EdgeDegreeEmbedding(
            sphere_channels=self.sphere_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            edge_channels_list=self.edge_channels_list,
            rescale_factor=5.0,
            mappingReduced=self.mappingReduced,
        )
        self.envelope = PolynomialEnvelope(exponent=5)

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.norm_type = norm_type
        self.act_type = act_type
        self.ff_type = ff_type

        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            self.blocks.append(
                eSCNMD_Block(
                    self.sphere_channels,
                    self.hidden_channels,
                    self.lmax,
                    self.mmax,
                    self.mappingReduced,
                    self.SO3_grid,
                    self.edge_channels_list,
                    norm_type=self.norm_type,
                    act_type=self.act_type,
                    ff_type=self.ff_type,
                )
            )

        self.norm = get_normalization_layer(
            self.norm_type, lmax=self.lmax, num_channels=self.sphere_channels
        )

        coefficient_index = self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
            self.lmax, self.mmax
        )
        self.register_buffer("coefficient_index", coefficient_index, persistent=False)

    # ---------------- helpers ----------------

    def csd_embedding(self, charge, spin, dataset):
        """Build the per-system CSD-mixed embedding."""
        chg_emb = self.charge_embedding(charge)
        spin_emb = self.spin_embedding(spin)
        if self.use_dataset_embedding:
            assert dataset is not None
            dataset_emb = self.dataset_embedding(dataset)
            return torch.nn.SiLU()(
                self.mix_csd(torch.cat((chg_emb, spin_emb, dataset_emb), dim=1))
            )
        return torch.nn.SiLU()(self.mix_csd(torch.cat((chg_emb, spin_emb), dim=1)))

    def balance_channels(
        self,
        x_message_prime: torch.Tensor,
        charge: torch.Tensor,
        spin: torch.Tensor,
        natoms: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.charge_channel_end > self.charge_channel_start:
            x_message_prime = balance_channels_batched(
                emb=x_message_prime,
                target=charge,
                natoms=natoms,
                batch=batch,
                start_idx=self.charge_channel_start,
                end_idx=self.charge_channel_end,
                target_offset=0.0,
            )
        if self.spin_channel_end > self.spin_channel_start:
            x_message_prime = balance_channels_batched(
                emb=x_message_prime,
                target=spin,
                natoms=natoms,
                batch=batch,
                start_idx=self.spin_channel_start,
                end_idx=self.spin_channel_end,
                target_offset=1.0,
            )
        return x_message_prime

    def _get_rotmat_and_wigner(self, edge_distance_vecs: torch.Tensor):
        Jd_buffers = [
            getattr(self, f"Jd_{l}").type(edge_distance_vecs.dtype)
            for l in range(self.lmax + 1)
        ]
        euler_angles = init_edge_rot_euler_angles(edge_distance_vecs)
        wigner = eulers_to_wigner(euler_angles, 0, self.lmax, Jd_buffers)
        wigner_inv = torch.transpose(wigner, 1, 2).contiguous()
        return wigner, wigner_inv

    # ---------------- forward ----------------

    def forward(self, data_dict: dict) -> dict:
        atomic_numbers = data_dict["atomic_numbers"].long()
        atomic_numbers_full = data_dict.get("atomic_numbers_full", atomic_numbers)
        batch = data_dict["batch"]
        batch_full = data_dict.get("batch_full", batch)
        edge_index = data_dict["edge_index"]
        edge_distance = data_dict["edge_distance"]
        edge_distance_vec = data_dict["edge_distance_vec"]
        natoms = data_dict["natoms"]
        nsystems = natoms.shape[0]

        # CSD mixed embedding (per system)
        csd_mixed_emb = self.csd_embedding(
            charge=data_dict["charge"],
            spin=data_dict["spin"],
            dataset=data_dict.get("dataset", None),
        )

        # MoLE coefficients (per system)
        self.set_MOLE_coefficients(
            atomic_numbers_full=atomic_numbers_full,
            batch_full=batch_full,
            csd_mixed_emb=csd_mixed_emb,
        )

        # Wigner D
        wigner, wigner_inv = self._get_rotmat_and_wigner(edge_distance_vec)
        coefficient_index = (
            self.coefficient_index if self.mmax != self.lmax else None
        )
        wigner, wigner_inv = prepare_wigner(
            wigner, wigner_inv, self.mappingReduced, coefficient_index
        )

        # Atom embedding + system embedding
        x_message = torch.zeros(
            atomic_numbers.shape[0],
            self.sph_feature_size,
            self.sphere_channels,
            device=edge_distance_vec.device,
            dtype=edge_distance_vec.dtype,
        )
        x_message[:, 0, :] = self.sphere_embedding(atomic_numbers)

        sys_node_embedding = csd_mixed_emb[batch]
        x_message[:, 0, :] = x_message[:, 0, :] + sys_node_embedding

        # MoLE sizes (atoms per system)
        self.set_MOLE_sizes(
            nsystems=csd_mixed_emb.shape[0],
            batch_full=batch_full,
            edge_index=edge_index,
        )
        self.log_MOLE_stats()

        # Edge feature: distance basis + atomic number embeddings (source, target)
        dist_scaled = edge_distance / self.cutoff
        edge_envelope = self.envelope(dist_scaled).reshape(-1, 1, 1)
        edge_distance_embedding = self.distance_expansion(edge_distance)
        source_embedding = self.source_embedding(atomic_numbers_full[edge_index[0]])
        target_embedding = self.target_embedding(atomic_numbers_full[edge_index[1]])
        x_edge = torch.cat(
            (edge_distance_embedding, source_embedding, target_embedding), dim=1
        )

        wigner_inv_envelope = wigner_inv * edge_envelope

        x_message = self.edge_degree_embedding(
            x_message, x_edge, edge_index, wigner_inv_envelope
        )

        # Message passing
        for i in range(self.num_layers):
            x_message = self.blocks[i](
                x_message,
                x_edge,
                edge_index,
                wigner,
                wigner_inv_envelope,
                sys_node_embedding=sys_node_embedding,
            )
            x_message = self.balance_channels(
                x_message,
                charge=data_dict["charge"],
                spin=data_dict["spin"],
                natoms=natoms,
                batch=batch,
            )

        x_message = self.norm(x_message)
        return {"node_embedding": x_message, "batch": batch}

    # ---------------- MoLE no-ops (overridden by MoeBackbone) ----------------

    def set_MOLE_coefficients(self, atomic_numbers_full, batch_full, csd_mixed_emb) -> None:
        return None

    def set_MOLE_sizes(self, nsystems, batch_full, edge_index) -> None:
        return None

    def log_MOLE_stats(self) -> None:
        return None


# ============================================================================
# Section 18: MoE backbone (eSCNMDMoeBackbone)
# ============================================================================

class eSCNMDMoeBackbone(eSCNMDBackbone, MOLEInterface):
    """eSCN-MD with MoLE / MoE expert routing on top of the base backbone."""

    def __init__(
        self,
        num_experts: int = 8,
        moe_dropout: float = 0.0,
        use_composition_embedding: bool = False,
        composition_dropout: float = 0.0,
        moe_expert_coefficient_norm: str = "softmax",
        layers_moe: Optional[List[int]] = None,
        moe_layer_type: str = "pytorch",
        moe_single: bool = False,
        moe_type: str = "so2",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.parent_kwargs = kwargs
        self.num_experts = num_experts
        if num_experts > 0:
            convert_model_to_MOLE_model(
                model=self,
                num_experts=num_experts,
                mole_dropout=moe_dropout,
                mole_expert_coefficient_norm=moe_expert_coefficient_norm,
                act=torch.nn.SiLU,
                layers_mole=layers_moe,
                use_composition_embedding=use_composition_embedding,
                composition_dropout=composition_dropout,
                mole_layer_type=moe_layer_type,
                mole_single=moe_single,
                mole_type=moe_type,
            )

    def set_MOLE_coefficients(self, atomic_numbers_full, batch_full, csd_mixed_emb) -> None:
        if self.num_experts == 0:
            return
        with torch.autocast(
            device_type=atomic_numbers_full.device.type, enabled=False
        ):
            embeddings = []
            if self.use_composition_embedding:
                effective_atomic_numbers_full = atomic_numbers_full
                effective_batch_full = batch_full
                if self.training and self.composition_dropout > 0.0:
                    mask = (
                        torch.rand_like(atomic_numbers_full, dtype=torch.float)
                        > self.composition_dropout
                    )
                    effective_atomic_numbers_full = atomic_numbers_full[mask]
                    effective_batch_full = batch_full[mask]
                composition_by_atom = self.composition_embedding(
                    effective_atomic_numbers_full
                )
                composition = composition_by_atom.new_zeros(
                    csd_mixed_emb.shape[0], self.sphere_channels
                ).index_reduce_(
                    0,
                    effective_batch_full,
                    composition_by_atom,
                    reduce="mean",
                    include_self=True,
                )
                embeddings.append(composition.unsqueeze(0))
            embeddings.append(csd_mixed_emb[None])
            stacked = torch.vstack(embeddings).transpose(0, 1).reshape(
                csd_mixed_emb.shape[0], -1
            )
            expert_mixing_coefficients_before_norm = self.routing_mlp(stacked)
            self.global_mole_tensors.expert_mixing_coefficients = (
                self.mole_expert_coefficient_norm(
                    self.mole_dropout(expert_mixing_coefficients_before_norm)
                )
            )

    def set_MOLE_sizes(self, nsystems, batch_full, edge_index) -> None:
        if self.num_experts == 0:
            return
        with torch.autocast(device_type=batch_full.device.type, enabled=False):
            mole_sizes = torch.zeros(
                nsystems,
                dtype=torch.int,
                device=batch_full[edge_index[1]].device,
            ).scatter_(0, batch_full[edge_index[1]], 1, reduce="add")
            self.global_mole_tensors.mole_sizes = mole_sizes.cpu()

    def log_MOLE_stats(self) -> None:
        # No-op (matplotlib path stripped from this port).
        return None


# ============================================================================
# Section 19: IANN adapter (UMA top-level model + GradientOutput)
# ============================================================================

class GradientOutput(torch.nn.Module):
    """IANN-style autograd output: forces from dE/d(edge_vectors), virial from dE/d(displacement)."""

    def __init__(
        self,
        grad_on_edge_diff: bool = True,
        grad_on_positions: bool = False,
        model_outputs: Optional[List[str]] = None,
        update_callback: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.grad_on_edge_diff = grad_on_edge_diff
        self.grad_on_positions = grad_on_positions
        self.update_callback = update_callback
        self.model_outputs = model_outputs if model_outputs is not None else ["forces"]

    def update_model_outputs(self, outputs: Union[List[str], str]):
        if isinstance(outputs, str):
            self.model_outputs.append(outputs)
        else:
            self.model_outputs.extend(outputs)
        if self.update_callback:
            self.update_callback()

    def forward(self, data: AtomsData, training: bool = True) -> AtomsData:
        if not self.grad_on_edge_diff:
            return data
        energy = data.energy
        edge_vectors = data.edge_vectors
        forces_dim = int(torch.sum(data.num_atoms))
        edge_indices = data.edge_indices
        assert energy is not None

        outputs_list = [energy]
        inputs_list = []
        grad_outputs_list = [torch.ones_like(energy, dtype=torch.float32)]

        compute_forces = "forces" in self.model_outputs
        compute_virial = "virial" in self.model_outputs
        compute_stress = "stress" in self.model_outputs

        if compute_forces:
            inputs_list.append(edge_vectors)

        displacement = data.displacement
        if displacement is not None and (compute_virial or compute_stress):
            inputs_list.append(displacement)

        if len(inputs_list) == 0:
            return data

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
            if dE_ddiff is not None:
                i_forces = torch.zeros(
                    (forces_dim, 3), device=edge_vectors.device, dtype=torch.float32
                )
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


class UMA(nn.Module):
    """Top-level IANN-friendly UMA model.

    Holds an ``eSCNMDMoeBackbone`` + an energy head + IANN-style
    ``GradientOutput``. Consumes IANN's ``AtomsData`` and returns ``AtomsData``
    with predictions filled in.

    Charge / spin / dataset are constructor kwargs (constants per training
    run), so the ``AtomsData`` schema does not need to change. The CSD
    embeddings and (optional) MoLE machinery are fully wired and trainable.
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cpu",
        num_channels: int = 128,
        num_layers: int = 4,
        norm_data: bool = False,
        norm_per_atom: bool = False,
        data_stddev: Union[float, List[float]] = 1.0,
        data_mean: Union[float, List[float]] = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )
        self.num_channels = num_channels
        self.num_layers = num_layers

        # Architecture kwargs
        self.lmax = int(kwargs.get("lmax", 2))
        self.mmax = int(kwargs.get("mmax", 2))
        self.cutoff = float(kwargs.get("cutoff", 5.0))
        self.edge_channels = int(kwargs.get("edge_channels", num_channels))
        self.num_distance_basis = int(kwargs.get("num_distance_basis", 512))
        self.hidden_channels = int(kwargs.get("hidden_channels", num_channels))
        self.norm_type = str(kwargs.get("norm_type", "rms_norm_sh"))
        self.act_type = str(kwargs.get("act_type", "gate"))
        self.ff_type = str(kwargs.get("ff_type", "grid"))
        self.grid_resolution = kwargs.get("grid_resolution", None)
        self.max_num_elements = int(kwargs.get("max_num_elements", 100))

        # CSD inputs (constants per training run)
        self.charge = int(kwargs.get("charge", 0))
        self.spin = int(kwargs.get("spin", 0))
        self.dataset_name = str(kwargs.get("dataset_name", "oc20"))
        self.use_dataset_embedding = bool(kwargs.get("use_dataset_embedding", True))
        dataset_names = kwargs.get(
            "dataset_names", ["oc20", "omol", "omat", "odac", "omc"]
        )
        if self.dataset_name not in dataset_names:
            dataset_names = list(dataset_names) + [self.dataset_name]
        self.dataset_mapping = {name: name for name in dataset_names}
        self.chg_spin_emb_type = str(kwargs.get("chg_spin_emb_type", "pos_emb"))

        # MoLE config
        self.num_experts = int(kwargs.get("num_experts", 0))
        self.moe_dropout = float(kwargs.get("moe_dropout", 0.0))
        self.use_composition_embedding = bool(
            kwargs.get("use_composition_embedding", False)
        )
        self.composition_dropout = float(kwargs.get("composition_dropout", 0.0))
        self.moe_expert_coefficient_norm = str(
            kwargs.get("moe_expert_coefficient_norm", "softmax")
        )
        self.moe_type = str(kwargs.get("moe_type", "so2"))
        self.moe_single = bool(kwargs.get("moe_single", False))
        self.layers_moe = kwargs.get("layers_moe", None)

        # Channel balancing
        self.charge_balanced_channels = list(
            kwargs.get("charge_balanced_channels", [])
        )
        self.spin_balanced_channels = list(
            kwargs.get("spin_balanced_channels", [])
        )

        # Output flags (set by Trainer based on loss weights)
        self.compute_forces = bool(kwargs.get("compute_forces", False))
        self.compute_stress = bool(kwargs.get("compute_stress", False))
        self.compute_virial = bool(kwargs.get("compute_virial", False))

        # Head selection. Default: MLP_Energy_Head + autograd forces/stress.
        self.head_type = str(kwargs.get("head_type", "mlp_energy"))

        # Build backbone (always MoeBackbone; degenerates to vanilla when num_experts=0)
        self.backbone = eSCNMDMoeBackbone(
            num_experts=self.num_experts,
            moe_dropout=self.moe_dropout,
            use_composition_embedding=self.use_composition_embedding,
            composition_dropout=self.composition_dropout,
            moe_expert_coefficient_norm=self.moe_expert_coefficient_norm,
            layers_moe=self.layers_moe,
            moe_layer_type="pytorch",
            moe_single=self.moe_single,
            moe_type=self.moe_type,
            # backbone kwargs
            max_num_elements=self.max_num_elements,
            sphere_channels=self.num_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            grid_resolution=self.grid_resolution,
            cutoff=self.cutoff,
            edge_channels=self.edge_channels,
            num_distance_basis=self.num_distance_basis,
            num_layers=self.num_layers,
            hidden_channels=self.hidden_channels,
            norm_type=self.norm_type,
            act_type=self.act_type,
            ff_type=self.ff_type,
            chg_spin_emb_type=self.chg_spin_emb_type,
            cs_emb_grad=bool(kwargs.get("cs_emb_grad", False)),
            dataset_emb_grad=bool(kwargs.get("dataset_emb_grad", False)),
            dataset_mapping=self.dataset_mapping,
            use_dataset_embedding=self.use_dataset_embedding,
            charge_balanced_channels=self.charge_balanced_channels,
            spin_balanced_channels=self.spin_balanced_channels,
        )

        # Output head (energy)
        if self.head_type == "mlp_energy":
            self.energy_head = MLP_Energy_Head(
                sphere_channels=self.num_channels,
                hidden_channels=self.hidden_channels,
                reduce="sum",
            )
        elif self.head_type == "linear_energy":
            self.energy_head = Linear_Energy_Head(
                sphere_channels=self.num_channels, reduce="sum"
            )
        else:
            raise ValueError(f"Unknown head_type {self.head_type}")

        self.energy_bias = nn.Parameter(torch.zeros(1))

        # Norm-data denorm (matches NequIP/EquiformerV2 / legacy uma.py behavior)
        self.norm_data = nn.Parameter(torch.tensor(norm_data), requires_grad=False)
        self.norm_per_atom = nn.Parameter(
            torch.tensor(norm_per_atom), requires_grad=False
        )
        self.data_stddev = nn.Parameter(
            torch.tensor(data_stddev), requires_grad=False
        )
        self.data_mean = nn.Parameter(torch.tensor(data_mean), requires_grad=False)

        # Optional autograd forces/stress hook
        outputs = [
            o for o in ("forces", "stress", "virial") if getattr(self, f"compute_{o}")
        ]
        self.gradient_output = (
            GradientOutput(model_outputs=outputs) if outputs else None
        )

    # ---------------- helpers ----------------

    def _apply_displacement(self, data: AtomsData) -> AtomsData:
        """Add a learnable strain (B,3,3) and re-derive edge_vectors for stress/virial autograd."""
        if data.image_indices is None:
            return data
        num_images = int(data.image_indices.max() + 1)
        displacement = torch.zeros(
            (num_images, 3, 3),
            dtype=data.edge_vectors.dtype,
            device=data.edge_vectors.device,
        ).requires_grad_()
        image_idx = data.image_indices[data.edge_indices[:, 0]]
        edge_vectors = data.edge_vectors + torch.bmm(
            displacement[image_idx], data.edge_vectors.unsqueeze(-1)
        ).squeeze(-1)
        return replace_properties(
            data, edge_vectors=edge_vectors, displacement=displacement
        )

    def _build_data_dict(self, data: AtomsData) -> dict:
        edge_vectors = data.edge_vectors
        edge_index = data.edge_indices.T.contiguous()
        atomic_numbers = data.atomic_numbers.long()
        natoms = data.num_atoms
        device = edge_vectors.device

        # When called directly by MLCalculator (single Atoms, not collated),
        # image_indices may be None - build it as all-zeros (one system).
        batch = data.image_indices
        if batch is None:
            batch = torch.repeat_interleave(
                torch.arange(natoms.shape[0], device=device), natoms.to(device)
            )
        nsys = natoms.shape[0]

        edge_distance = torch.norm(edge_vectors, dim=1)
        charge_t = torch.full(
            (nsys,), self.charge, dtype=torch.long, device=device
        )
        spin_t = torch.full(
            (nsys,), self.spin, dtype=torch.long, device=device
        )
        dataset = [self.dataset_name] * nsys

        return {
            "atomic_numbers": atomic_numbers,
            "atomic_numbers_full": atomic_numbers,
            "batch": batch,
            "batch_full": batch,
            "edge_index": edge_index,
            "edge_distance": edge_distance,
            "edge_distance_vec": edge_vectors,
            "natoms": natoms,
            "charge": charge_t,
            "spin": spin_t,
            "dataset": dataset,
        }

    # ---------------- forward ----------------

    def forward(self, data: AtomsData) -> AtomsData:
        if self.compute_stress or self.compute_virial:
            data = self._apply_displacement(data)

        data_dict = self._build_data_dict(data)
        emb = self.backbone(data_dict)

        # Energy head
        nsys = int(data.num_atoms.shape[0])
        natoms = data.num_atoms.to(emb["node_embedding"].device)
        energy = self.energy_head(emb, num_systems=nsys, natoms=natoms)
        energy = energy.to(emb["node_embedding"].dtype) + self.energy_bias

        # Norm-data denorm (post-sum, matches NequIP/EquiformerV2 conventions)
        if bool(self.norm_data.item()):
            if bool(self.norm_per_atom.item()):
                energy = self.data_stddev * energy + (
                    natoms.to(energy.dtype) * self.data_mean
                )
            else:
                energy = self.data_stddev * energy + self.data_mean

        data = replace_properties(data, energy=energy)

        if self.gradient_output is not None:
            data = self.gradient_output(data, training=self.training)
        return data
