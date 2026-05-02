import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import copy
from typing import Optional, List, Union, Callable, Literal
from e3nn import o3
from e3nn.o3 import FromS2Grid, ToS2Grid

from iann.data import AtomsData, replace_properties
import iann

# === Section 1: Radial Utilities ===

@torch.jit.script
def gaussian(x: torch.Tensor, mean, std) -> torch.Tensor:
    a = (2 * math.pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class PolynomialEnvelope(torch.nn.Module):
    def __init__(self, exponent: int = 5) -> None:
        super().__init__()
        assert exponent > 0
        self.p: float = float(exponent)
        self.a: float = -(self.p + 1) * (self.p + 2) / 2
        self.b: float = self.p * (self.p + 2)
        self.c: float = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        env_val = 1 + (d_scaled**self.p) * (
            self.a + d_scaled * (self.b + self.c * d_scaled)
        )
        return torch.where(d_scaled < 1, env_val, 0)

class GaussianSmearing(torch.nn.Module):
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
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset, persistent=False)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class RadialMLP(nn.Module):
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

# === Section 2: SO3 Utilities ===

class CoefficientMapping(torch.nn.Module):
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
                self.register_buffer(f"coefficient_idx_l{l}_m{m}", mask_indices, persistent=False)

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
            mask = torch.bitwise_and(self.l_harmonic.le(lmax), self.m_harmonic.le(mmax))
            indices = torch.arange(len(mask), device=mask.device)
            return torch.masked_select(indices, mask)
        else:
            temp = self.prepare_coefficient_idx()
            return temp[lmax][mmax]

class SO3_Grid(torch.nn.Module):
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

        to_grid = ToS2Grid(self.lmax, (self.lat_resolution, self.long_resolution), normalization=normalization)
        to_grid_mat = torch.einsum("mbi, am -> bai", to_grid.shb, to_grid.sha).detach()
        if rescale and lmax != mmax:
            for lval in range(lmax + 1):
                if lval <= mmax: continue
                start_idx = lval**2
                length = 2 * lval + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                to_grid_mat[:, :, start_idx : (start_idx + length)] *= rescale_factor
        to_grid_mat = to_grid_mat[:, :, self.mapping.coefficient_idx(self.lmax, self.mmax)]

        from_grid = FromS2Grid((self.lat_resolution, self.long_resolution), self.lmax, normalization=normalization)
        from_grid_mat = torch.einsum("am, mbi -> bai", from_grid.sha, from_grid.shb).detach()
        if rescale and lmax != mmax:
            for lval in range(lmax + 1):
                if lval <= mmax: continue
                start_idx = lval**2
                length = 2 * lval + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                from_grid_mat[:, :, start_idx : (start_idx + length)] *= rescale_factor
        from_grid_mat = from_grid_mat[:, :, self.mapping.coefficient_idx(self.lmax, self.mmax)]

        self.register_buffer("to_grid_mat", to_grid_mat, persistent=False)
        self.register_buffer("from_grid_mat", from_grid_mat, persistent=False)

    def to_grid(self, embedding, lmax, mmax):
        to_grid_mat = self.to_grid_mat[:, :, self.mapping.coefficient_idx(lmax, mmax)]
        return torch.einsum("bai, zic -> zbac", to_grid_mat, embedding)

    def from_grid(self, grid, lmax, mmax):
        from_grid_mat = self.from_grid_mat[:, :, self.mapping.coefficient_idx(lmax, mmax)]
        return torch.einsum("bai, zbac -> zic", from_grid_mat, grid)

# === Section 3: Rotation Utilities ===

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
    xyz = torch.nn.functional.normalize(edge_distance_vec).clamp(-1.0, 1.0)
    x, y, z = torch.split(xyz, 1, dim=1)
    beta = Safeacos.apply(y.squeeze(-1))
    alpha = Safeatan2.apply(x.squeeze(-1), z.squeeze(-1))
    gamma = torch.zeros_like(alpha) # UMA often uses zero gamma or random, zero is fine
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
    wigner = torch.zeros(len(alpha), size, size, device=alpha.device, dtype=alpha.dtype)
    start = 0
    for l in range(start_lmax, end_lmax + 1):
        block = wigner_D(l, alpha, beta, gamma, Jd)
        end = start + block.size()[1]
        wigner[:, start:end, start:end] = block
        start = end
    return wigner

# === Section 4: Normalization ===

def get_l_to_all_m_expand_index(lmax: int):
    expand_index = torch.zeros([(lmax + 1) ** 2]).long()
    for lval in range(lmax + 1):
        start_idx = lval**2
        length = 2 * lval + 1
        expand_index[start_idx : (start_idx + length)] = lval
    return expand_index

class EquivariantRMSNormArraySphericalHarmonicsV2(nn.Module):
    def __init__(self, lmax, num_channels, eps=1e-5, affine=True, normalization="component", centering=True, std_balance_degrees=True):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.centering = centering
        self.std_balance_degrees = std_balance_degrees
        if affine:
            self.affine_weight = nn.Parameter(torch.ones((self.lmax + 1), self.num_channels))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_channels)) if centering else None
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)
        self.normalization = normalization
        self.register_buffer("expand_index", get_l_to_all_m_expand_index(self.lmax), persistent=False)
        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2, 1)
            for lval in range(self.lmax + 1):
                start_idx = lval**2
                length = 2 * lval + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = 1.0 / length
            balance_degree_weight /= (self.lmax + 1)
            self.register_buffer("balance_degree_weight", balance_degree_weight, persistent=False)

    def forward(self, node_input):
        feature = node_input
        if self.centering:
            feature_l0 = feature.narrow(1, 0, 1)
            feature_l0_mean = feature_l0.mean(dim=2, keepdim=True)
            feature_l0 = feature_l0 - feature_l0_mean
            feature = torch.cat((feature_l0, feature.narrow(1, 1, feature.shape[1] - 1)), dim=1)
        if self.normalization == "norm":
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)
        elif self.normalization == "component":
            if self.std_balance_degrees:
                feature_norm = torch.einsum("nic, ia -> nac", feature.pow(2), self.balance_degree_weight)
            else:
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)
        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)
        feature_norm = (feature_norm + self.eps).pow(-0.5)
        if self.affine:
            weight = torch.index_select(self.affine_weight.view(1, self.lmax + 1, self.num_channels), dim=1, index=self.expand_index)
            feature_norm = feature_norm * weight
        out = feature * feature_norm
        if self.affine and self.centering:
            out[:, 0:1, :] = out.narrow(1, 0, 1) + self.affine_bias.view(1, 1, self.num_channels)
        return out

def get_normalization_layer(norm_type, lmax, num_channels, eps=1e-5, affine=True, normalization="component"):
    if norm_type == "rms_norm_sh":
        return EquivariantRMSNormArraySphericalHarmonicsV2(lmax, num_channels, eps, affine, normalization)
    raise ValueError(f"Unknown norm_type: {norm_type}")

# === Section 5: SO3 Linear ===

class SO3_Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, lmax) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax
        self.weight = torch.nn.Parameter(torch.randn((self.lmax + 1), out_features, in_features))
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        self.register_buffer("expand_index", get_l_to_all_m_expand_index(lmax), persistent=False)

    def forward(self, input_embedding):
        weight = torch.index_select(self.weight, dim=0, index=self.expand_index)
        out = torch.einsum("bmi, moi -> bmo", input_embedding, weight).contiguous()
        out[:, 0:1, :] = out.narrow(1, 0, 1) + self.bias.view(1, 1, self.out_features)
        return out

# === Section 6: SO2 Convolution ===

class SO2_m_Conv_Block(torch.nn.Module):
    def __init__(self, m, sphere_channels, m_output_channels, lmax, mmax) -> None:
        super().__init__()
        self.m = m
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax
        num_coefficents = self.lmax - m + 1
        num_channels = num_coefficents * self.sphere_channels
        self.out_channels_half = self.m_output_channels * num_coefficents
        self.fc = nn.Linear(num_channels, 2 * self.out_channels_half, bias=False)
        self.fc.weight.data.mul_(1 / math.sqrt(2))

    def forward(self, x_m):
        W1, W2 = self.fc.weight.split(self.out_channels_half, dim=0)
        W = torch.cat([torch.cat([W1, -W2], dim=1), torch.cat([W2, W1], dim=1)], dim=0)
        out = x_m.flatten(1) @ W.T
        return out.view(-1, 2, self.out_channels_half // self.m_output_channels, self.m_output_channels).unbind(1)

class SO2_Convolution(torch.nn.Module):
    def __init__(self, sphere_channels, m_output_channels, lmax, mmax, mappingReduced, internal_weights=True, edge_channels_list=None, extra_m0_output_channels=None) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced
        self.internal_weights = internal_weights
        self.extra_m0_output_channels = extra_m0_output_channels
        num_channels_m0 = (lmax + 1) * sphere_channels
        m0_out = self.m_output_channels * (lmax + 1)
        if extra_m0_output_channels: m0_out += extra_m0_output_channels
        self.fc_m0 = nn.Linear(num_channels_m0, m0_out)
        self.so2_m_conv = nn.ModuleList([SO2_m_Conv_Block(m, sphere_channels, m_output_channels, lmax, mmax) for m in range(1, mmax + 1)])
        self.rad_func = RadialMLP(edge_channels_list + [num_channels_m0 + sum(m.fc.in_features for m in self.so2_m_conv)]) if not internal_weights else None
        self.m_split_sizes = [mappingReduced.m_size[0]] + (torch.tensor(mappingReduced.m_size[1:]) * 2).tolist()
        self.edge_split_sizes = [self.fc_m0.in_features] + [m.fc.in_features for m in self.so2_m_conv]

    def forward(self, x, x_edge=None):
        if self.rad_func: x_edge = self.rad_func(x_edge)
        x_by_m = x.split(self.m_split_sizes, dim=1)
        if x_edge is not None: x_edge_by_m = x_edge.split(self.edge_split_sizes, dim=1)
        num_edges = x.shape[0]
        x_0 = x_by_m[0].view(num_edges, -1)
        if x_edge is not None: x_0 = x_0 * x_edge_by_m[0]
        x_0 = self.fc_m0(x_0)
        if self.extra_m0_output_channels:
            x_0_extra, x_0 = x_0.split((self.extra_m0_output_channels, x_0.shape[1] - self.extra_m0_output_channels), -1)
        out = [x_0.view(num_edges, -1, self.m_output_channels)]
        for m in range(1, self.mmax + 1):
            x_m = x_by_m[m].view(num_edges, 2, -1)
            if x_edge is not None: x_m = x_m * x_edge_by_m[m].unsqueeze(1)
            out.extend(self.so2_m_conv[m-1](x_m))
        out = torch.cat(out, dim=1)
        return (out, x_0_extra) if self.extra_m0_output_channels else out

# === Section 7: Activation ===

class ScaledSiLU(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.scale_factor = 1.6791767923989418
    def forward(self, inputs):
        return F.silu(inputs, inplace=self.inplace) * self.scale_factor

class GateActivation(torch.nn.Module):
    def __init__(self, lmax, mmax, num_channels, m_prime=True) -> None:
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
                expand_index[start_idx : (start_idx + length)] = torch.cat([torch.arange(mval - 1, lmax), torch.arange(mval - 1, lmax)])
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
        gating_scalars = self.gate_act(gating_scalars).view(gating_scalars.shape[0], self.lmax, self.num_channels)
        gating_scalars = torch.index_select(gating_scalars, dim=1, index=self.expand_index)
        scalars, vectors = input_tensors.split((1, input_tensors.shape[1] - 1), 1)
        return torch.cat((self.scalar_act(scalars), vectors * gating_scalars), dim=1)

class S2Activation_M(torch.nn.Module):
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
        output_scalars = self.scalar_act(input_scalars).reshape(input_scalars.shape[0], 1, -1)
        output_tensors = self.s2_act(input_tensors)
        return torch.cat((output_scalars, output_tensors.narrow(1, 1, output_tensors.shape[1] - 1)), dim=1)

# === Section 8: Embeddings ===

class EdgeDegreeEmbedding(torch.nn.Module):
    def __init__(self, sphere_channels, lmax, mmax, edge_channels_list, rescale_factor, mappingReduced):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced
        self.m_0_num_coefficients = self.mappingReduced.m_size[0]
        self.rad_func = RadialMLP(edge_channels_list + [self.m_0_num_coefficients * self.sphere_channels])
        self.rescale_factor = rescale_factor

    def forward(self, x, x_edge, edge_index, wigner_inv_envelope):
        radial = self.rad_func(x_edge).reshape(-1, self.m_0_num_coefficients, self.sphere_channels)
        wigner_inv_m0 = wigner_inv_envelope[:, :, :self.m_0_num_coefficients]
        x_edge_embedding = torch.bmm(wigner_inv_m0, radial)
        return x.index_add(0, edge_index[1], x_edge_embedding / self.rescale_factor)

# === Section 9: Backend Operations (General) ===

def prepare_wigner(wigner, wigner_inv, mappingReduced, coefficient_index=None):
    if coefficient_index is not None:
        wigner = wigner.index_select(1, coefficient_index)
        wigner_inv = wigner_inv.index_select(2, coefficient_index)
    wigner = torch.einsum("mk,nkj->nmj", mappingReduced.to_m.to(wigner.dtype), wigner)
    wigner_inv = torch.einsum("njk,mk->njm", wigner_inv, mappingReduced.to_m.to(wigner_inv.dtype))
    return wigner, wigner_inv

def node_to_edge_wigner_permute(x_full, edge_index, wigner):
    x_source = x_full[edge_index[0]]
    x_target = x_full[edge_index[1]]
    x_message = torch.cat((x_source, x_target), dim=2)
    return torch.bmm(wigner, x_message)

def permute_wigner_inv_edge_to_node(x_message, wigner_inv, edge_index, num_nodes):
    x_rotated = torch.bmm(wigner_inv, x_message)
    new_embedding = torch.zeros((num_nodes,) + x_rotated.shape[1:], dtype=x_rotated.dtype, device=x_rotated.device)
    new_embedding.index_add_(0, edge_index[1], x_rotated)
    return new_embedding

# === Section 10: Blocks ===

class Edgewise(nn.Module):
    def __init__(self, sphere_channels, hidden_channels, lmax, mmax, mappingReduced, SO3_grid, edge_channels_list, act_type='gate'):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.mappingReduced = mappingReduced
        
        if act_type == 'gate':
            self.act = GateActivation(lmax, mmax, hidden_channels, m_prime=True)
            extra_m0 = lmax * hidden_channels
        else:
            self.act = SeparableS2Activation_M(lmax, mmax, SO3_grid, mappingReduced.to_m)
            extra_m0 = hidden_channels
            
        self.so2_conv_1 = SO2_Convolution(2 * sphere_channels, hidden_channels, lmax, mmax, mappingReduced, internal_weights=False, edge_channels_list=edge_channels_list, extra_m0_output_channels=extra_m0)
        self.so2_conv_2 = SO2_Convolution(hidden_channels, sphere_channels, lmax, mmax, mappingReduced, internal_weights=True)

    def forward(self, x_full, x_edge, edge_index, wigner, wigner_inv):
        x_message = node_to_edge_wigner_permute(x_full, edge_index, wigner)
        x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)
        x_message = self.act(x_0_gating, x_message)
        x_message = self.so2_conv_2(x_message)
        return permute_wigner_inv_edge_to_node(x_message, wigner_inv, edge_index, x_full.shape[0])

class GridAtomwise(nn.Module):
    def __init__(self, sphere_channels, hidden_channels, lmax, mmax, mappingReduced, SO3_grid):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.SO3_grid = SO3_grid
        self.linear_1 = SO3_Linear(sphere_channels, hidden_channels, lmax)
        self.act = SeparableS2Activation_M(lmax, mmax, SO3_grid, mappingReduced.to_m)
        self.linear_2 = SO3_Linear(hidden_channels, sphere_channels, lmax)

    def forward(self, x):
        x = self.linear_1(x)
        x_scalars = x.narrow(1, 0, 1).squeeze(1)
        x = self.act(x_scalars, x)
        return self.linear_2(x)

class eSCNMD_Block(nn.Module):
    def __init__(self, sphere_channels, hidden_channels, lmax, mmax, mappingReduced, SO3_grid, edge_channels_list, act_type='gate'):
        super().__init__()
        self.norm_1 = get_normalization_layer('rms_norm_sh', lmax, sphere_channels)
        self.edge_wise = Edgewise(sphere_channels, hidden_channels, lmax, mmax, mappingReduced, SO3_grid, edge_channels_list, act_type=act_type)
        self.norm_2 = get_normalization_layer('rms_norm_sh', lmax, sphere_channels)
        self.atom_wise = GridAtomwise(sphere_channels, hidden_channels, lmax, mmax, mappingReduced, SO3_grid)
        self.res_scale_edge = nn.Parameter(torch.ones(1))
        self.res_scale_atom = nn.Parameter(torch.ones(1))

    def forward(self, x, x_edge, edge_index, wigner, wigner_inv):
        x_res = x
        x = self.norm_1(x)
        x = x_res + self.res_scale_edge * self.edge_wise(x, x_edge, edge_index, wigner, wigner_inv)
        
        x_res = x
        x = self.norm_2(x)
        x = x_res + self.res_scale_atom * self.atom_wise(x)
        return x

# === Section 11: Output Utilities ===

class GradientOutput(torch.nn.Module):
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

    def forward(self, data: AtomsData, training: bool=True) -> AtomsData:
        if self.grad_on_edge_diff:
            energy = data.energy
            edge_vectors = data.edge_vectors
            forces_dim = int(torch.sum(data.num_atoms))
            edge_indices = data.edge_indices
            assert energy is not None
            
            outputs_list = [energy]
            inputs_list = []
            grad_outputs_list = [torch.ones_like(energy, dtype=torch.float32)]
            
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
                    allow_unused=True
                )
                
                idx = 0
                if compute_forces:
                    dE_ddiff = grads[idx]
                    idx += 1
                    if dE_ddiff is not None:
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

# === Section 12: Main UMA Wrapper ===

class UMA(nn.Module):
    def __init__(self, device='cpu', num_channels=128, num_layers=4, norm_data=False, norm_per_atom=False, data_stddev=1.0, data_mean=0.0, **kwargs):
        super().__init__()
        self.device = torch.device(device)
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.lmax = kwargs.get('lmax', 2)
        self.mmax = kwargs.get('mmax', 2)
        self.cutoff = kwargs.get('cutoff', 5.0)
        self.edge_channels = kwargs.get('edge_channels', 128)
        self.num_basis = kwargs.get('num_distance_basis', 512)
        self.hidden_channels = kwargs.get('hidden_channels', num_channels)
        self.norm_type = kwargs.get('norm_type', 'rms_norm_sh')
        self.grid_res = kwargs.get('grid_resolution', None)
        self.compute_forces = kwargs.get('compute_forces', False)
        self.compute_stress = kwargs.get('compute_stress', False)
        self.compute_virial = kwargs.get('compute_virial', False)

        self.mapping = CoefficientMapping(self.lmax, self.mmax)
        self.grid = SO3_Grid(self.lmax, self.mmax, resolution=self.grid_res)
        self.sphere_emb = nn.Embedding(119, num_channels)
        self.dist_exp = GaussianSmearing(0.0, self.cutoff, self.num_basis)
        self.envelope = PolynomialEnvelope(5)
        
        self.source_emb = nn.Embedding(119, self.edge_channels)
        self.target_emb = nn.Embedding(119, self.edge_channels)
        edge_in = self.num_basis + 2 * self.edge_channels
        
        self.edge_degree_emb = EdgeDegreeEmbedding(num_channels, self.lmax, self.mmax, [edge_in], 1.0, self.mapping)
        
        self.blocks = nn.ModuleList([eSCNMD_Block(num_channels, self.hidden_channels, self.lmax, self.mmax, self.mapping, self.grid, [edge_in], act_type=kwargs.get('act_type', 'gate')) for _ in range(num_layers)])
        self.norm = get_normalization_layer(self.norm_type, self.lmax, num_channels)
        
        self.energy_block = nn.Sequential(nn.Linear(num_channels, num_channels), nn.SiLU(), nn.Linear(num_channels, 1))
        self.energy_bias = nn.Parameter(torch.zeros(1))
        
        Jd_path = os.path.join(iann.__path__[0], "data", "Jd.pt")
        Jd_list = torch.load(Jd_path, weights_only=True)
        for l in range(self.lmax + 1):
            self.register_buffer(f"Jd_{l}", Jd_list[l])

        self.norm_data = nn.Parameter(torch.tensor(norm_data), requires_grad=False)
        self.norm_per_atom = nn.Parameter(torch.tensor(norm_per_atom), requires_grad=False)
        self.data_stddev = nn.Parameter(torch.tensor(data_stddev), requires_grad=False)
        self.data_mean = nn.Parameter(torch.tensor(data_mean), requires_grad=False)

        self.gradient_output = GradientOutput(model_outputs=[o for o in ['forces', 'stress', 'virial'] if getattr(self, f'compute_{o}')]) if (self.compute_forces or self.compute_stress or self.compute_virial) else None

    def _apply_displacement(self, data: AtomsData) -> AtomsData:
        if data.image_indices is None: return data
        num_images = int(data.image_indices.max() + 1)
        displacement = torch.zeros((num_images, 3, 3), dtype=data.edge_vectors.dtype, device=data.edge_vectors.device).requires_grad_()
        image_idx = data.image_indices[data.edge_indices[:, 0]]
        edge_vectors = data.edge_vectors + torch.bmm(displacement[image_idx], data.edge_vectors.unsqueeze(-1)).squeeze(-1)
        return replace_properties(data, edge_vectors=edge_vectors, displacement=displacement)

    def forward(self, data: AtomsData):
        if self.compute_stress or self.compute_virial: data = self._apply_displacement(data)
        
        edge_vectors = data.edge_vectors
        edge_index = data.edge_indices.T
        atomic_numbers = data.atomic_numbers.long()
        image_indices = data.image_indices
        num_atoms = len(atomic_numbers)
        
        edge_dist = torch.norm(edge_vectors, dim=1)
        eulers = init_edge_rot_euler_angles(edge_vectors)
        Jd = [getattr(self, f"Jd_{l}") for l in range(self.lmax + 1)]
        wigner = eulers_to_wigner(eulers, 0, self.lmax, Jd)
        wigner_inv = wigner.transpose(1, 2).contiguous()
        
        wigner, wigner_inv = prepare_wigner(wigner, wigner_inv, self.mapping)
        
        x = torch.zeros((num_atoms, (self.lmax+1)**2, self.num_channels), device=edge_vectors.device, dtype=edge_vectors.dtype)
        x[:, 0, :] = self.sphere_emb(atomic_numbers)
        
        x_edge = torch.cat([self.dist_exp(edge_dist), self.source_emb(atomic_numbers[edge_index[0]]), self.target_emb(atomic_numbers[edge_index[1]])], dim=-1)
        env = self.envelope(edge_dist / self.cutoff).view(-1, 1, 1)
        wigner_inv_env = wigner_inv * env
        
        x = self.edge_degree_emb(x, x_edge, edge_index, wigner_inv_env)
        
        for block in self.blocks:
            x = block(x, x_edge, edge_index, wigner, wigner_inv_env)
        
        x = self.norm(x)
        node_energy = self.energy_block(x[:, 0, :])
        
        energy = torch.zeros(data.num_atoms.shape[0], device=node_energy.device, dtype=node_energy.dtype)
        energy.index_add_(0, image_indices, node_energy.view(-1))
        energy = energy + self.energy_bias
        
        if self.norm_data:
            energy = self.data_stddev * energy + (data.num_atoms.to(energy.dtype) * self.data_mean if self.norm_per_atom else self.data_mean)
        
        data = replace_properties(data, energy=energy)
        if self.gradient_output: data = self.gradient_output(data, training=self.training)
        return data



