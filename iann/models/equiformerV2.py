import torch
from torch import nn
import copy, os
from e3nn import o3
import math
import torch_geometric
from typing import List, Optional
from iann.data.data import AtomsData, replace_properties
import iann


class SO3_Grid(torch.nn.Module):
    """
    Helper functions for grid representation of the irreps

    Args:
        lmax (int):   Maximum degree of the spherical harmonics
        mmax (int):   Maximum order of the spherical harmonics
    """

    def __init__(
        self,
        lmax: int,
        mmax: int,
        normalization='integral', 
        resolution=None,
        device=None,
    ):
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

        self.mapping = CoefficientMappingModule([self.lmax], [self.lmax], device)
        to_grid = o3.ToS2Grid(
            self.lmax,
            (self.lat_resolution, self.long_resolution),
            normalization=normalization, 
            device=device,
        )
        to_grid_mat = torch.einsum("mbi, am -> bai", to_grid.shb, to_grid.sha).detach()
        # rescale based on mmax
        if lmax != mmax:
            for l in range(lmax + 1):
                if l <= mmax:
                    continue
                start_idx = l ** 2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                to_grid_mat[:, :, start_idx : (start_idx + length)] = to_grid_mat[:, :, start_idx : (start_idx + length)] * rescale_factor
        to_grid_mat = to_grid_mat[:, :, self.mapping.coefficient_idx(self.lmax, self.mmax)]

        from_grid = o3.FromS2Grid(
            (self.lat_resolution, self.long_resolution),
            self.lmax,
            normalization=normalization, #normalization="integral",
            device=device,
        )
        from_grid_mat = torch.einsum("am, mbi -> bai", from_grid.sha, from_grid.shb).detach()
        # rescale based on mmax
        if lmax != mmax:
            for l in range(lmax + 1):
                if l <= mmax:
                    continue
                start_idx = l ** 2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                from_grid_mat[:, :, start_idx : (start_idx + length)] = from_grid_mat[:, :, start_idx : (start_idx + length)] * rescale_factor
        from_grid_mat = from_grid_mat[:, :, self.mapping.coefficient_idx(self.lmax, self.mmax)]

        # save tensors and they will be moved to GPU
        self.register_buffer('to_grid_mat',   to_grid_mat, persistent=False)
        self.register_buffer('from_grid_mat', from_grid_mat, persistent=False)


    # Compute matrices to transform irreps to grid
    @torch.jit.export
    def get_to_grid_mat(self, device: torch.device):
        return self.to_grid_mat.to(device)


    # Compute matrices to transform grid to irreps
    @torch.jit.export
    def get_from_grid_mat(self, device: torch.device):
        return self.from_grid_mat.to(device)


    # Compute grid from irreps representation
    @torch.jit.export
    def to_grid(self, embedding: torch.Tensor, lmax: int, mmax: int):
        indices = self.mapping.coefficient_idx(lmax, mmax)
        # Ensure indices are on the same device as the tensor being indexed
        indices = indices.to(self.to_grid_mat.device)
        to_grid_mat = self.to_grid_mat[:, :, indices]
        # Ensure both tensors are on the same device before einsum
        to_grid_mat = to_grid_mat.to(embedding.device)
        grid = torch.einsum("bai, zic -> zbac", to_grid_mat, embedding)
        return grid


    # Compute irreps from grid representation
    @torch.jit.export
    def from_grid(self, grid: torch.Tensor, lmax: int, mmax: int):
        indices = self.mapping.coefficient_idx(lmax, mmax)
        # Ensure indices are on the same device as the tensor being indexed
        indices = indices.to(self.from_grid_mat.device)
        from_grid_mat = self.from_grid_mat[:, :, indices]
        # Ensure both tensors are on the same device before einsum
        from_grid_mat = from_grid_mat.to(grid.device)
        embedding = torch.einsum("bai, zbac -> zic", from_grid_mat, grid)
        return embedding


class CoefficientMappingModule(torch.nn.Module):
    """
    Helper module for coefficients used to reshape l <--> m and to get coefficients of specific degree or order

    Args:
        lmax_list (list:int):   List of maximum degree of the spherical harmonics
        mmax_list (list:int):   List of maximum order of the spherical harmonics
    """

    def __init__(
        self,
        lmax_list: list[int],
        mmax_list: list[int],
        device: str,
    ):
        super().__init__()

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)

        # Temporarily use `cpu` as device and this will be overwritten.
        self.device = device
        
        # Compute the degree (l) and order (m) for each entry of the embedding
        l_harmonic = torch.zeros(0, device=self.device).long()
        m_harmonic = torch.zeros(0, device=self.device).long()
        m_complex  = torch.zeros(0, device=self.device).long()

        res_size = torch.zeros([self.num_resolutions], device=self.device).long()

        offset = 0
        for i in range(self.num_resolutions):
            for l in range(0, self.lmax_list[i] + 1):
                mmax = min(self.mmax_list[i], l)
                m = torch.arange(-mmax, mmax + 1, device=self.device).long()
                m_complex = torch.cat([m_complex, m], dim=0)
                m_harmonic = torch.cat(
                    [m_harmonic, torch.abs(m).long()], dim=0
                )
                l_harmonic = torch.cat(
                    [l_harmonic, m.fill_(l).long()], dim=0
                )
            res_size[i] = len(l_harmonic) - offset
            offset = len(l_harmonic)

        num_coefficients = len(l_harmonic)
        # `self.to_m` moves m components from different L to contiguous index
        to_m = torch.zeros([num_coefficients, num_coefficients], device=self.device)
        m_size = torch.zeros([max(self.mmax_list) + 1], device=self.device).long()

        # The following is implemented poorly - very slow. It only gets called
        # a few times so haven't optimized.
        offset = 0
        for m in range(max(self.mmax_list) + 1):
            idx_r, idx_i = self.complex_idx(m, -1, m_complex, l_harmonic)

            for idx_out, idx_in in enumerate(idx_r):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_r)

            m_size[m] = int(len(idx_r))

            for idx_out, idx_in in enumerate(idx_i):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_i)

        to_m = to_m.detach()

        # save tensors and they will be moved to GPU
        self.register_buffer('l_harmonic', l_harmonic, persistent=False )
        self.register_buffer('m_harmonic', m_harmonic, persistent=False)
        self.register_buffer('m_complex',  m_complex, persistent=False)
        self.register_buffer('res_size',   res_size, persistent=False)
        self.register_buffer('to_m',       to_m, persistent=False)
        self.register_buffer('m_size',     m_size, persistent=False)

        # for caching the output of `coefficient_idx`
        self.lmax_cache = -3 # -3 is initialized value
        self.mmax_cache = -3 # -3 is initialized value
        self.mask_indices_cache = torch.zeros(0, dtype=torch.long, device=self.device)  # Initialize as empty tensor
        self.rotate_inv_rescale_cache = torch.zeros(0, dtype=torch.float, device=self.device)

    def complex_idx(self, m, lmax, m_complex, l_harmonic):
        '''
            Add `m_complex` and `l_harmonic` to the input arguments 
            since we cannot use `self.m_complex`. 

            Return mask containing coefficients of order m (real and imaginary parts)
        '''
        if lmax == -1:
            lmax = max(self.lmax_list)

        indices = torch.arange(len(l_harmonic), device=l_harmonic.device)
        # Real part
        mask_r = torch.bitwise_and(
            l_harmonic.le(lmax), m_complex.eq(m)
        )
        mask_idx_r = torch.masked_select(indices, mask_r)

        mask_idx_i = torch.tensor([], device=l_harmonic.device).long()
        # Imaginary part
        if m != 0:
            mask_i = torch.bitwise_and(
                l_harmonic.le(lmax), m_complex.eq(-m)
            )
            mask_idx_i = torch.masked_select(indices, mask_i)

        return mask_idx_r, mask_idx_i


    def coefficient_idx(self, lmax: int, mmax: int):
        """
            Return mask containing coefficients less than or equal to degree (l) and order (m)
        """

        if (self.lmax_cache == -3) or (self.mmax_cache == -3):
            if (self.lmax_cache == lmax) and (self.mmax_cache == mmax):
                if self.mask_indices_cache.numel() != 0:
                    return self.mask_indices_cache

        mask = torch.bitwise_and(
            self.l_harmonic.le(lmax), self.m_harmonic.le(mmax)
        )
        # Use the device of the existing tensors instead of self.device
        indices = torch.arange(len(mask), device=mask.device)
        mask_indices = torch.masked_select(indices, mask)
        self.lmax_cache, self.mmax_cache = lmax, mmax
        self.mask_indices_cache = mask_indices
        return self.mask_indices_cache
    

    def get_rotate_inv_rescale(self, lmax: int, mmax: int):
        """
            Return the re-scaling for rotating back to original frame
            this is required since we only use a subset of m components for SO(2) convolution
        """

        if (self.lmax_cache is not None) and (self.mmax_cache is not None):
            if (self.lmax_cache == lmax) and (self.mmax_cache == mmax):
                if self.rotate_inv_rescale_cache.numel() != 0:
                    return self.rotate_inv_rescale_cache
        
        if self.mask_indices_cache.numel() == 0:
            self.coefficient_idx(lmax, mmax)
        
        size = int((lmax + 1) ** 2)
        # Use the device of the existing tensors instead of self.device
        rotate_inv_rescale = torch.ones((1, size, size), device=self.mask_indices_cache.device)
        for l in range(lmax + 1):
            if l <= mmax:
                continue
            start_idx = int(l ** 2)
            length = int(2 * l + 1)
            rescale_factor = math.sqrt(length / (2 * mmax + 1))
            rotate_inv_rescale[:, start_idx : (start_idx + length), start_idx : (start_idx + length)] = rescale_factor
        rotate_inv_rescale = rotate_inv_rescale[:, :, self.mask_indices_cache]        
        self.rotate_inv_rescale_cache = rotate_inv_rescale
        return self.rotate_inv_rescale_cache

    
    def __repr__(self):
        return f"{self.__class__.__name__}(lmax_list={self.lmax_list}, mmax_list={self.mmax_list})"

def get_normalization_layer(norm_type, lmax, num_channels, eps=1e-5, affine=True, normalization='component'):
    assert norm_type in ['layer_norm', 'layer_norm_sh', 'rms_norm_sh']
    if norm_type == 'layer_norm':
        norm_class = EquivariantLayerNormArray
    elif norm_type == 'layer_norm_sh':
        norm_class = EquivariantLayerNormArraySphericalHarmonics
    elif norm_type == 'rms_norm_sh':
        norm_class = EquivariantRMSNormArraySphericalHarmonicsV2
    else:
        raise ValueError
    return norm_class(lmax, num_channels, eps, affine, normalization)

def get_l_to_all_m_expand_index(lmax):
    expand_index = torch.zeros([(lmax + 1) ** 2]).long()
    for l in range(lmax + 1):
        start_idx = l ** 2
        length = 2 * l + 1
        expand_index[start_idx : (start_idx + length)] = l
    return expand_index

class SO3_Rotation(torch.nn.Module):
    """
    Helper functions for Wigner-D rotations

    Args:
        lmax_list (list:int):   List of maximum degree of the spherical harmonics
    """

    def __init__(
        self,
        lmax,
        device,
    ):
        super().__init__()
        self.lmax = lmax
        self.mapping = CoefficientMappingModule([self.lmax], [self.lmax], device)
        self.device = device
        self._Jd = torch.load(
            os.path.join(iann.__path__[0], "data", "Jd.pt"),
            weights_only=True,
        )

    @torch.jit.export
    def set_wigner(self, rot_mat3x3: torch.Tensor):
        wigner = self.RotationToWignerDMatrix(rot_mat3x3, 0, self.lmax)
        wigner_inv = torch.transpose(wigner, 1, 2).contiguous()
        wigner = wigner.detach()
        wigner_inv = wigner_inv.detach()
        return wigner, wigner_inv

    @torch.jit.export
    def rotate(self, embedding: torch.Tensor, out_lmax: int, out_mmax: int, wigner: torch.Tensor):
        """
            Rotate the embedding by the rotation matrix
        """
        out_mask = self.mapping.coefficient_idx(out_lmax, out_mmax)
        wigner = wigner[:, out_mask, :]
        return torch.bmm(wigner, embedding)

    @torch.jit.export
    def rotate_inv(self, embedding: torch.Tensor, in_lmax: int, in_mmax: int, wigner_inv: torch.Tensor):
        """
            Rotate the embedding by the inverse of the rotation matrix 
        """
        in_mask = self.mapping.coefficient_idx(in_lmax, in_mmax)
        wigner_inv = wigner_inv[:, :, in_mask]
        wigner_inv_rescale = self.mapping.get_rotate_inv_rescale(in_lmax, in_mmax)
        wigner_inv = wigner_inv * wigner_inv_rescale
        if wigner_inv.shape[0] != embedding.shape[0]:
            raise RuntimeError(f"rotate_inv(): Batch mismatch: wigner_inv {wigner_inv.shape}, embedding {embedding.shape}")
        return torch.bmm(wigner_inv, embedding)
    

    @torch.jit.export
    def wigner_D(self, l: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor):
        """
            Compute the Wigner D matrix

            In 0.5.0, e3nn shifted to torch.matrix_exp which is significantly slower:
            https://github.com/e3nn/e3nn/blob/0.5.0/e3nn/o3/_wigner.py#L92

            Borrowed from e3nn @ 0.4.0:
            https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L10

            _Jd is a list of tensors of shape (2l+1, 2l+1)
        """
        if not l < len(self._Jd):
            raise NotImplementedError(
                f"wigner D maximum l implemented is {len(self._Jd) - 1}, send us an email to ask for more"
            )

        alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
        J = self._Jd[l].to(dtype=alpha.dtype, device=alpha.device)
        Xa = self._z_rot_mat(alpha, l)
        Xb = self._z_rot_mat(beta, l)
        Xc = self._z_rot_mat(gamma, l)
        return Xa @ J @ Xb @ J @ Xc

    @torch.jit.export
    def _z_rot_mat(self, angle: torch.Tensor, l: int) -> torch.Tensor:
        shape = angle.shape
        device = angle.device
        dtype = angle.dtype

        size = 2 * l + 1
        M = torch.zeros(list(shape) + [size, size], dtype=dtype, device=device)

        cos_a = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)
        sin_a = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)

        for i in range(size):
            for j in range(size):
                if i == j:
                    M[..., i, j] = cos_a.squeeze(-1).squeeze(-1)
                if i + j == 2 * l:
                    M[..., i, j] = sin_a.squeeze(-1).squeeze(-1)

        return M

    @torch.jit.export
    def RotationToWignerDMatrix(self, edge_rot_mat: torch.Tensor, start_lmax: int, end_lmax: int):
        """
            Compute Wigner matrices from rotation matrix
        """
        x = edge_rot_mat @ torch.tensor([0.0, 1.0, 0.0], dtype=edge_rot_mat.dtype, device=edge_rot_mat.device)
        alpha, beta = o3.xyz_to_angles(x)
        R = (
            o3.angles_to_matrix(
                alpha, beta, torch.zeros_like(alpha)
            ).transpose(-1, -2)
            @ edge_rot_mat
        )
        gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])

        size = int((end_lmax + 1) ** 2 - (start_lmax) ** 2)
        wigner = torch.zeros(int(len(alpha)), size, size, device=self.device)
        start = 0
        for lmax in range(start_lmax, end_lmax + 1):
            block = self.wigner_D(lmax, alpha, beta, gamma)
            end = start + block.size()[1]
            wigner[:, start:end, start:end] = block
            start = end

        return wigner.detach()


class SO3_Embedding(nn.Module):
    """
    Helper functions for performing operations on irreps embedding

    Args:
        length (int):           Batch size
        lmax_list (list[int]):   List of maximum degree of the spherical harmonics
        num_channels (int):     Number of channels
        device:                 Device of the output
        dtype:                  type of the output tensors
    """

    def __init__(
        self,
        length: int,
        lmax_list: List[int],
        num_channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.device = device
        self.dtype = dtype
        self.num_resolutions = len(lmax_list)

        self.num_coefficients = 0
        for i in range(self.num_resolutions):
            self.num_coefficients = self.num_coefficients + int(
                (lmax_list[i] + 1) ** 2
            )
        self.register_buffer("dummy_buffer", torch.empty(0), persistent=False)
        embedding = torch.zeros(
            length,
            self.num_coefficients,
            self.num_channels,
            device=self.dummy_buffer.device,
            dtype=self.dummy_buffer.dtype,
        )
        self.set_embedding(embedding)
        self.set_lmax_mmax(lmax_list, lmax_list.copy())

    @torch.jit.export
    def set_embedding(self, embedding):
        self.length = len(embedding)
        self.embedding = embedding

    @torch.jit.export
    def set_lmax_mmax(self, lmax_list: list[int], mmax_list: list[int]):
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list

    @torch.jit.export
    def _expand_edge(self, edge_idx):
        embedding = self.embedding[edge_idx]
        self.set_embedding(embedding)

    @torch.jit.export
    def _reduce_edge(self, edge_idx: torch.Tensor, num_nodes: int):
        new_embedding = torch.zeros(
            num_nodes,
            self.num_coefficients,
            self.num_channels,
            device=self.dummy_buffer.device,
            dtype=self.dummy_buffer.dtype,
        )
        new_embedding.index_add_(0, edge_idx, self.embedding)
        self.set_embedding(new_embedding)

    @torch.jit.export
    def _m_primary(self, mapping: CoefficientMappingModule):
        self.embedding = torch.einsum("nac, ba -> nbc", self.embedding, mapping.to_m)

    @torch.jit.export
    def _l_primary(self, mapping: CoefficientMappingModule):
        self.embedding = torch.einsum("nac, ab -> nbc", self.embedding, mapping.to_m)

    @torch.jit.export
    def _rotate(self, SO3_rotation: List[SO3_Rotation], lmax_list: List[int], mmax_list: List[int], edge_rot_mat: torch.Tensor):
        if self.num_resolutions == 1:
            wigner, _ = SO3_rotation[0].set_wigner(edge_rot_mat)
            embedding_rotate = SO3_rotation[0].rotate(self.embedding, lmax_list[0], mmax_list[0], wigner)
        else:
            offset = 0
            embedding_rotate = torch.tensor([], device=self.dummy_buffer.device, dtype=self.dummy_buffer.dtype)
            for i in range(self.num_resolutions):
                num_coefficients = int((self.lmax_list[i] + 1) ** 2)
                embedding_i = self.embedding[:, offset : offset + num_coefficients]
                wigner, _ = SO3_rotation[i].set_wigner(edge_rot_mat)
                embedding_rotate = torch.cat([
                        embedding_rotate,
                        SO3_rotation[i].rotate(embedding_i, lmax_list[i], mmax_list[i], wigner)],
                    dim=1)
                offset = offset + num_coefficients

        self.embedding = embedding_rotate
        self.set_lmax_mmax(lmax_list.copy(), mmax_list.copy())

    @torch.jit.export
    def _rotate_inv(self, SO3_rotation: List[SO3_Rotation], mappingReduced: CoefficientMappingModule, edge_rot_mat: torch.Tensor):
        if self.num_resolutions == 1:
            wigner, wigner_inv = SO3_rotation[0].set_wigner(edge_rot_mat)
            embedding_rotate = SO3_rotation[0].rotate_inv(self.embedding, self.lmax_list[0], self.mmax_list[0], wigner_inv)
        else:
            offset = 0
            embedding_rotate = torch.tensor([], device=self.dummy_buffer.device, dtype=self.dummy_buffer.dtype)
            for i in range(self.num_resolutions):
                num_coefficients = mappingReduced.res_size[i]
                embedding_i = self.embedding[:, offset : offset + num_coefficients]
                wigner, wigner_inv = SO3_rotation[i].set_wigner(edge_rot_mat)
                embedding_rotate = torch.cat([
                        embedding_rotate,
                        SO3_rotation[i].rotate_inv(embedding_i, self.lmax_list[i], self.mmax_list[i], wigner_inv)],
                    dim=1)
                offset = offset + num_coefficients
        self.embedding = embedding_rotate

        # Assume mmax = lmax when rotating back
        for i in range(self.num_resolutions):
            self.mmax_list[i] = int(self.lmax_list[i])
        self.set_lmax_mmax(self.lmax_list, self.mmax_list)

    @torch.jit.export
    def to_grid(self, SO3_grid: List[SO3_Grid], lmax: int = -1):
        if lmax == -1:
            lmax = max(self.lmax_list)
        SO3_grid = list(SO3_grid)
        idx = self._get_grid_index(lmax, lmax, lmax)
        to_grid_mat_lmax = SO3_grid[idx].get_to_grid_mat(self.dummy_buffer.device)
        grid_mapping     = SO3_grid[idx].mapping

        offset = 0
        # Initialize x_grid on the same device as the input tensors
        x_grid = torch.tensor([], device=self.dummy_buffer.device)

        for i in range(self.num_resolutions):
            num_coefficients = int((self.lmax_list[i] + 1) ** 2)
            if self.num_resolutions == 1:
                x_res = self.embedding
            else:
                x_res = self.embedding[:, offset : offset + num_coefficients].contiguous()
            indices = grid_mapping.coefficient_idx(self.lmax_list[i], self.lmax_list[i])
            to_grid_mat = to_grid_mat_lmax[:, :, indices]
            x_grid = torch.cat([x_grid, torch.einsum("bai, zic -> zbac", to_grid_mat, x_res)], dim=3)
            offset = offset + num_coefficients

        return x_grid
    
    @torch.jit.export
    def _get_grid_index(self, l: int, m: int, max_l: int) -> int:
        return l * (max_l + 1) + m

    @torch.jit.export
    def _from_grid(self, x_grid: torch.Tensor, SO3_grid: list[SO3_Grid], lmax: int =-1):
        if lmax ==-1:
            lmax = max(self.lmax_list)
        
        idx = self._get_grid_index(lmax, lmax, lmax)
        from_grid_mat_lmax = SO3_grid[idx].get_from_grid_mat(self.dummy_buffer.device)
        grid_mapping       = SO3_grid[idx].mapping
        offset = 0
        offset_channel = 0
        for i in range(self.num_resolutions):
            indices = grid_mapping.coefficient_idx(self.lmax_list[i], self.lmax_list[i])
            # Ensure indices are on the same device as the tensor being indexed
            indices = indices.to(self.dummy_buffer.device)
            from_grid_mat = from_grid_mat_lmax[:, :, indices]
            if self.num_resolutions == 1:
                temp = x_grid
            else:
                temp = x_grid[:, :, :, offset_channel : offset_channel + self.num_channels]
            # Ensure both tensors are on the same device before einsum
            from_grid_mat = from_grid_mat.to(self.dummy_buffer.device)
            x_res = torch.einsum("bai, zbac -> zic", from_grid_mat, temp)
            # Ensure x_res is on the same device as self.embedding before assignment
            x_res = x_res.to(self.dummy_buffer.device)
            num_coefficients = int((self.lmax_list[i] + 1) ** 2)
            
            if self.num_resolutions == 1:
                self.embedding = x_res
            else: 
                self.embedding[:, offset : offset + num_coefficients] = x_res
            
            offset = offset + num_coefficients
            offset_channel = offset_channel + self.num_channels

def init_edge_rot_mat(edge_diff):
    edge_vec_0 = edge_diff
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

    # Make sure the atoms are far enough apart
    if torch.min(edge_vec_0_distance) < 0.0001:
        print(
            "Error edge_vec_0_distance: {}".format(
                torch.min(edge_vec_0_distance)
            )
        )
        
    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

    edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
    edge_vec_2 = edge_vec_2 / (
        torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1)
    )
    # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
    # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
    edge_vec_2b = edge_vec_2.clone()
    edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.clone()
    edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    edge_vec_2c[:, 2] = edge_vec_2[:, 1]
    vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(
        -1, 1
    )
    vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(
        -1, 1
    )

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(
        torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2
    )
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(
        torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2
    )

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
    # Check the vectors aren't aligned
    assert torch.max(vec_dot) < 0.99

    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / (
        torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True))
    )
    norm_z = norm_z / (
        torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1)
    )
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / (
        torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True))
    )

    # Construct the 3D rotation matrix
    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

    return edge_rot_mat.detach()

class GaussianSmearing(torch.nn.Module):
    """
        Gaussian smearing function, different encodings for the atom distance embeddings 
    """
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

class RadialFunction(nn.Module):
    '''
        Contruct a radial function (linear layers + layer normalization + SiLU) given a list of channels
    '''
    def __init__(self, channels_list):
        super().__init__()
        modules = []
        input_channels = channels_list[0]
        for i in range(len(channels_list)):
            if i == 0:
                continue
            if isinstance(channels_list[i], torch.Tensor):
                channels_list[i] = channels_list[i].item()
            modules.append(nn.Linear(input_channels, channels_list[i], bias=True))
            input_channels = channels_list[i]
            
            if i == len(channels_list) - 1:
                break
            
            modules.append(nn.LayerNorm(channels_list[i]))
            modules.append(torch.nn.SiLU())
        
        self.net = nn.Sequential(*modules)

        
    def forward(self, inputs):
        return self.net(inputs)
        

class EdgeDegreeEmbedding(torch.nn.Module):
    """

    Args:
        atom_channels (int):      Number of spherical channels
        
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        
        SO3_rotation (list:SO3_Rotation): Class to calculate Wigner-D matrices and rotate embeddings
        mappingReduced (CoefficientMappingModule): Class to convert l and m indices once node embedding is rotated
        
        max_num_elements (int):     Maximum number of atomic numbers
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                        The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features

        rescale_factor (float):     Rescale the sum aggregation
    """

    def __init__(
        self,
        atom_channels: int,
        
        lmax_list: list[int],
        mmax_list: list[int],
        
        SO3_rotation: list[SO3_Rotation],
        mappingReduced: CoefficientMappingModule,

        max_num_elements: int,
        edge_channels_list: list[int],
        use_atom_edge_embedding: bool,
        
        rescale_factor: float
    ):
        super(EdgeDegreeEmbedding, self).__init__()
        self.atom_channels = atom_channels
        self.lmax_list: list[int] = lmax_list
        self.mmax_list: list[int] = mmax_list
        self.num_resolutions = len(self.lmax_list)
        self.SO3_rotation = SO3_rotation
        self.mappingReduced = mappingReduced
        
        self.m_0_num_coefficients = self.mappingReduced.m_size[0] 
        self.m_all_num_coefficents = len(self.mappingReduced.l_harmonic)

        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
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

        # Embedding function of distance
        self.edge_channels_list.append(self.m_0_num_coefficients * self.atom_channels)
        self.rad_func = RadialFunction(self.edge_channels_list)

        self.rescale_factor = rescale_factor
        self.x_edge_embedding = SO3_Embedding(
            0, 
            self.lmax_list.copy(), 
            self.atom_channels, 
            device=self.m_0_num_coefficients.device, 
            dtype=self.m_0_num_coefficients.dtype
        )
        self.m_0_num_coefficients = self.m_0_num_coefficients.item() # convert tensor to static integer


    def forward(
        self,
        atomic_numbers: torch.Tensor,
        edge_dist: torch.Tensor,
        edge_idx: torch.Tensor,
        edge_rot_mat: torch.Tensor
    ):    
        
        if self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_idx[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_idx[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            x_edge = torch.cat((edge_dist, source_embedding, target_embedding), dim=1)
        else:
            x_edge = edge_dist

        x_edge_m_0 = self.rad_func(x_edge)
        
        x_edge_m_0 = x_edge_m_0.reshape(-1, self.m_0_num_coefficients, self.atom_channels)
        x_edge_m_pad = torch.zeros((
            x_edge_m_0.shape[0], 
            (self.m_all_num_coefficents - self.m_0_num_coefficients), 
            self.atom_channels), 
            device=x_edge_m_0.device)
        x_edge_m_all = torch.cat((x_edge_m_0, x_edge_m_pad), dim=1)
        
        self.x_edge_embedding.set_embedding(x_edge_m_all)
        self.x_edge_embedding.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # Reshape the spherical harmonics based on l (degree)
        self.x_edge_embedding._l_primary(self.mappingReduced)
        
        # Rotate back the irreps
        self.x_edge_embedding._rotate_inv(list(self.SO3_rotation), self.mappingReduced, edge_rot_mat)

        # Compute the sum of the incoming neighboring messages for each target node
        self.x_edge_embedding._reduce_edge(edge_idx[1], atomic_numbers.shape[0])
        self.x_edge_embedding.embedding = self.x_edge_embedding.embedding / self.rescale_factor

        return self.x_edge_embedding


class EquivariantLayerNormArray(nn.Module):
    
    def __init__(self, lmax, num_channels, eps=1e-5, affine=True, normalization='component'):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(lmax + 1, num_channels))
            self.affine_bias   = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization


    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps})"


    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, node_input):
        '''
            Assume input is of shape [N, atom_basis, C]
        '''
        
        out = []
        
        for l in range(self.lmax + 1):
            start_idx = l ** 2
            length = 2 * l + 1
            
            feature = node_input.narrow(1, start_idx, length)
            
            # For scalars, first compute and subtract the mean
            if l == 0:
                feature_mean = torch.mean(feature, dim=2, keepdim=True)
                feature = feature - feature_mean
                
            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == 'norm':
                feature_norm = feature.pow(2).sum(dim=1, keepdim=True)      # [N, 1, C]
            elif self.normalization == 'component':
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)     # [N, 1, C]
            else:
                # Either raise an error or define a safe default
                feature_norm = torch.ones_like(feature[:, :1, :])
                raise ValueError(f"Unknown normalization type: {self.normalization}")
            
            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)    # [N, 1, 1]
            feature_norm = (feature_norm + self.eps).pow(-0.5)
            
            if self.affine:
                weight = self.affine_weight.narrow(0, l, 1)     # [1, C]
                weight = weight.view(1, 1, -1)                  # [1, 1, C]
                feature_norm = feature_norm * weight            # [N, 1, C]
            
            feature = feature * feature_norm
            
            if self.affine and l == 0: 
                bias = self.affine_bias
                bias = bias.view(1, 1, -1)
                feature = feature + bias
            
            out.append(feature)
        
        out = torch.cat(out, dim=1)
        
        return out 


class EquivariantLayerNormArraySphericalHarmonics(nn.Module):
    '''
        1. Normalize over L = 0.
        2. Normalize across all m components from degrees L > 0.
        3. Do not normalize separately for different L (L > 0).
    '''
    def __init__(self, lmax, num_channels, eps=1e-5, affine=True, normalization='component', std_balance_degrees=True):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.std_balance_degrees = std_balance_degrees
        
        # for L = 0
        self.norm_l0 = torch.nn.LayerNorm(self.num_channels, eps=self.eps, elementwise_affine=self.affine)

        # for L > 0
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.lmax, self.num_channels))
        else:
            self.register_parameter('affine_weight', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2 - 1, 1)
            for l in range(1, self.lmax + 1):
                start_idx = l ** 2 - 1
                length = 2 * l + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = (1.0 / length)
            balance_degree_weight = balance_degree_weight / self.lmax
            self.register_buffer('balance_degree_weight', balance_degree_weight, persistent=False)
        else:
            self.balance_degree_weight = None


    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, std_balance_degrees={self.std_balance_degrees})"


    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, node_input):
        '''
            Assume input is of shape [N, atom_basis, C]
        '''
        
        out = []

        # for L = 0
        feature = node_input.narrow(1, 0, 1)
        feature = self.norm_l0(feature)
        out.append(feature)

        # for L > 0
        if self.lmax > 0:
            num_m_components = (self.lmax + 1) ** 2
            feature = node_input.narrow(1, 1, num_m_components - 1)

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == 'norm':
                assert not self.std_balance_degrees
                feature_norm = feature.pow(2).sum(dim=1, keepdim=True)      # [N, 1, C]
            elif self.normalization == 'component':
                if self.std_balance_degrees:
                    feature_norm = feature.pow(2)                               # [N, (L_max + 1)**2 - 1, C], without L = 0
                    feature_norm = torch.einsum('nic, ia -> nac', feature_norm, self.balance_degree_weight) # [N, 1, C]
                else:
                    feature_norm = feature.pow(2).mean(dim=1, keepdim=True)     # [N, 1, C]
            else:
                # Either raise an error or define a safe default
                feature_norm = torch.ones_like(feature[:, :1, :])
                raise ValueError(f"Unknown normalization type: {self.normalization}")
            
            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)    # [N, 1, 1]
            feature_norm = (feature_norm + self.eps).pow(-0.5)

            for l in range(1, self.lmax + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                feature = node_input.narrow(1, start_idx, length)       # [N, (2L + 1), C]
                if self.affine:
                    weight = self.affine_weight.narrow(0, (l - 1), 1)       # [1, C]
                    weight = weight.view(1, 1, -1)                          # [1, 1, C]
                    feature_scale = feature_norm * weight                   # [N, 1, C]
                else:
                    feature_scale = feature_norm
                feature = feature * feature_scale
                out.append(feature)
            
        out = torch.cat(out, dim=1)
        return out

    
class EquivariantRMSNormArraySphericalHarmonics(nn.Module):
    '''
        1. Normalize across all m components from degrees L >= 0.
    '''
    def __init__(self, lmax, num_channels, eps=1e-5, affine=True, normalization='component'):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        # for L >= 0
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones((self.lmax + 1), self.num_channels))
        else:
            self.register_parameter('affine_weight', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization


    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps})"


    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, node_input):
        '''
            Assume input is of shape [N, atom_basis, C]
        '''
        
        out = []

        # for L >= 0
        feature = node_input    
        if self.normalization == 'norm':
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)      # [N, 1, C]
        elif self.normalization == 'component':
            feature_norm = feature.pow(2).mean(dim=1, keepdim=True)     # [N, 1, C]
        else:
            # Either raise an error or define a safe default
            feature_norm = torch.ones_like(feature[:, :1, :])
            raise ValueError(f"Unknown normalization type: {self.normalization}")
            
        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)    # [N, 1, 1]
        feature_norm = (feature_norm + self.eps).pow(-0.5)

        for l in range(0, self.lmax + 1):
            start_idx = l ** 2
            length = 2 * l + 1
            feature = node_input.narrow(1, start_idx, length)       # [N, (2L + 1), C]
            if self.affine:
                weight = self.affine_weight.narrow(0, l, 1)         # [1, C]
                weight = weight.view(1, 1, -1)                      # [1, 1, C]
                feature_scale = feature_norm * weight               # [N, 1, C]
            else:
                feature_scale = feature_norm
            feature = feature * feature_scale
            out.append(feature)
            
        out = torch.cat(out, dim=1)
        return out

        
class EquivariantRMSNormArraySphericalHarmonicsV2(nn.Module):
    '''
        1. Normalize across all m components from degrees L >= 0.
        2. Expand weights and multiply with normalized feature to prevent slicing and concatenation.
    '''
    def __init__(self, lmax, num_channels, eps=1e-5, affine=True, normalization='component', centering=True, std_balance_degrees=True):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.centering = centering
        self.std_balance_degrees = std_balance_degrees
        
        # for L >= 0
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones((self.lmax + 1), self.num_channels))
            if self.centering:
                self.affine_bias = nn.Parameter(torch.zeros(self.num_channels))
            else:
                self.register_parameter('affine_bias', None)
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization

        expand_index = get_l_to_all_m_expand_index(self.lmax)
        self.register_buffer('expand_index', expand_index, persistent=False)

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2, 1)
            for l in range(self.lmax + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = (1.0 / length)
            balance_degree_weight = balance_degree_weight / (self.lmax + 1)
            self.register_buffer('balance_degree_weight', balance_degree_weight, persistent=False)
        else:
            self.balance_degree_weight = None


    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, centering={self.centering}, std_balance_degrees={self.std_balance_degrees})"


    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, node_input):
        '''
            Assume input is of shape [N, atom_basis, C]
        '''
        
        feature = node_input    
        
        if self.centering:
            feature_l0 = feature.narrow(1, 0, 1)
            feature_l0_mean = feature_l0.mean(dim=2, keepdim=True) # [N, 1, 1]
            feature_l0 = feature_l0 - feature_l0_mean
            feature = torch.cat((feature_l0, feature.narrow(1, 1, feature.shape[1] - 1)), dim=1)
            
        # for L >= 0
        if self.normalization == 'norm':
            assert not self.std_balance_degrees
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)      # [N, 1, C]
        elif self.normalization == 'component':
            if self.std_balance_degrees:
                feature_norm = feature.pow(2)                               # [N, (L_max + 1)**2, C]
                feature_norm = torch.einsum('nic, ia -> nac', feature_norm, self.balance_degree_weight) # [N, 1, C]
            else:
                feature_norm = torch.ones_like(feature[:, :1, :])
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)     # [N, 1, C]
        else:
            # Either raise an error or define a safe default
            feature_norm = torch.ones_like(feature[:, :1, :])
            raise ValueError(f"Unknown normalization type: {self.normalization}")
            
        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)    # [N, 1, 1]
        feature_norm = (feature_norm + self.eps).pow(-0.5)

        if self.affine:
            weight = self.affine_weight.view(1, (self.lmax + 1), self.num_channels)     # [1, L_max + 1, C]
            weight = torch.index_select(weight, dim=1, index=self.expand_index)         # [1, (L_max + 1)**2, C]
            feature_norm = feature_norm * weight                                        # [N, (L_max + 1)**2, C]
        
        out = feature * feature_norm

        if self.affine and self.centering:
            out[:, 0:1, :] = out.narrow(1, 0, 1) + self.affine_bias.view(1, 1, self.num_channels)

        return out


class EquivariantDegreeLayerScale(nn.Module):
    '''
        1. Similar to Layer Scale used in CaiT (Going Deeper With Image Transformers (ICCV'21)), we scale the output of both attention and FFN. 
        2. For degree L > 0, we scale down the square root of 2 * L, which is to emulate halving the number of channels when using higher L. 
    '''
    def __init__(self, lmax, num_channels, scale_factor=2.0):
        super().__init__()
        
        self.lmax = lmax
        self.num_channels = num_channels
        self.scale_factor = scale_factor

        self.affine_weight = nn.Parameter(torch.ones(1, (self.lmax + 1), self.num_channels))
        for l in range(1, self.lmax + 1):
            self.affine_weight.data[0, l, :].mul_(1.0 / math.sqrt(self.scale_factor * l))        
        expand_index = get_l_to_all_m_expand_index(self.lmax)
        self.register_buffer('expand_index', expand_index, persistent=False)


    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, scale_factor={self.scale_factor})"

    
    def forward(self, node_input):
        weight = torch.index_select(self.affine_weight, dim=1, index=self.expand_index)     # [1, (L_max + 1)**2, C]
        node_input = node_input * weight                                                    # [N, (L_max + 1)**2, C]
        return node_input

class SO2_m_Convolution(torch.nn.Module):
    """
    SO(2) Conv: Perform an SO(2) convolution on features corresponding to +- m

    Args:
        m (int):                    Order of the spherical harmonic coefficients
        atom_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
    """
    def __init__(
        self,
        m, 
        atom_channels,
        m_output_channels,
        lmax_list, 
        mmax_list
    ):
        super(SO2_m_Convolution, self).__init__()
        
        self.m = m
        self.atom_channels = atom_channels
        self.m_output_channels = m_output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(self.lmax_list)

        num_channels = 0
        for i in range(self.num_resolutions):
            num_coefficents = 0
            if self.mmax_list[i] >= self.m:
                num_coefficents = self.lmax_list[i] - self.m + 1
            num_channels = num_channels + num_coefficents * self.atom_channels
        assert num_channels > 0
        self.out_features = 2 * self.m_output_channels * (num_channels // self.atom_channels)
        if isinstance(self.out_features, torch.Tensor):
            self.out_features = self.out_features.item()
        assert isinstance(self.out_features, int)
        self.fc = torch.nn.Linear(num_channels, self.out_features, bias=False)
        self.fc.weight.data.mul_(1 / math.sqrt(2))


    def forward(self, x_m):
        x_m = self.fc(x_m)
        x_r = x_m.narrow(2, 0, self.out_features // 2)
        x_i = x_m.narrow(2, self.out_features // 2, self.out_features // 2)
        x_m_r = x_r.narrow(1, 0, 1) - x_i.narrow(1, 1, 1) #x_r[:, 0] - x_i[:, 1]
        x_m_i = x_r.narrow(1, 1, 1) + x_i.narrow(1, 0, 1) #x_r[:, 1] + x_i[:, 0]
        x_out = torch.cat((x_m_r, x_m_i), dim=1)
        
        return x_out


class SO2_Convolution(torch.nn.Module):
    """
    SO(2) Block: Perform SO(2) convolutions for all m (orders)

    Args:
        atom_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        mappingReduced (CoefficientMappingModule): Used to extract a subset of m components
        internal_weights (bool):    If True, not using radial function to multiply inputs features
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
        extra_m0_output_channels (int): If not None, return `out_embedding` (SO3_Embedding) and `extra_m0_features` (Tensor).
    """
    def __init__(
        self,
        atom_channels,
        m_output_channels,
        lmax_list,
        mmax_list,
        mappingReduced,
        internal_weights=True,
        edge_channels_list=None,
        extra_m0_output_channels=None
    ):
        super(SO2_Convolution, self).__init__()
        self.atom_channels = atom_channels
        self.m_output_channels = m_output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.mappingReduced = mappingReduced
        self.num_resolutions = len(lmax_list)
        self.internal_weights = internal_weights
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.extra_m0_output_channels = extra_m0_output_channels

        num_channels_rad = 0    # for radial function

        num_channels_m0 = 0
        for i in range(self.num_resolutions):
            num_coefficients = self.lmax_list[i] + 1
            num_channels_m0 = num_channels_m0 + num_coefficients * self.atom_channels

        # SO(2) convolution for m = 0
        m0_output_channels = self.m_output_channels * (num_channels_m0 // self.atom_channels)
        if self.extra_m0_output_channels is not None:
            m0_output_channels = m0_output_channels + self.extra_m0_output_channels

        # Store output features as a constant integer
        self.fc_m0_out_features: int = m0_output_channels
        if isinstance(num_channels_m0, torch.Tensor):
            num_channels_m0 = num_channels_m0.item()
        if isinstance(m0_output_channels, torch.Tensor):
            m0_output_channels = m0_output_channels.item()
        assert isinstance(num_channels_m0, int)
        assert isinstance(m0_output_channels, int)
        self.fc_m0 = torch.nn.Linear(num_channels_m0, m0_output_channels)
        num_channels_rad = num_channels_rad + self.fc_m0.in_features
        
        # SO(2) convolution for non-zero m
        self.so2_m_conv = nn.ModuleList()
        for m in range(1, max(self.mmax_list) + 1):
            self.so2_m_conv.append(
                SO2_m_Convolution(
                    m, 
                    self.atom_channels,
                    self.m_output_channels,
                    self.lmax_list, 
                    self.mmax_list,
                )
            )
            num_channels_rad = num_channels_rad + self.so2_m_conv[-1].fc.in_features

        # Embedding function of distance
        self.rad_func = None
        if not self.internal_weights:
            assert self.edge_channels_list is not None
            self.edge_channels_list.append(int(num_channels_rad))
            self.rad_func = RadialFunction(self.edge_channels_list)
        
        self.register_buffer("dummy_buffer", torch.empty(0), persistent=False)
        self.out_embedding = SO3_Embedding(
            0, 
            self.lmax_list.copy(), 
            self.m_output_channels, 
            device=self.dummy_buffer.device, 
            dtype=self.dummy_buffer.dtype
        )


    def forward(self, x: SO3_Embedding, x_edge: torch.Tensor) -> tuple[SO3_Embedding, Optional[torch.Tensor]]:
        num_edges = len(x_edge)
        out = []

        # Reshape the spherical harmonics based on m (order)
        x._m_primary(self.mappingReduced)

        # radial function
        if self.rad_func is not None:
            x_edge = self.rad_func(x_edge)
        offset_rad = 0

        # Compute m=0 coefficients separately since they only have real values (no imaginary)
        x_0 = x.embedding.narrow(1, 0, self.mappingReduced.m_size[0])
        x_0 = x_0.reshape(num_edges, -1)
        if self.rad_func is not None:
            x_edge_0 = x_edge.narrow(1, 0, self.fc_m0.in_features)
            x_0 = x_0 * x_edge_0
        x_0 = self.fc_m0(x_0)

        x_0_extra = None
        # extract extra m0 features 
        if self.extra_m0_output_channels is not None:
            x_0_extra = x_0.narrow(-1, 0, self.extra_m0_output_channels)
            x_0 = x_0.narrow(-1, self.extra_m0_output_channels, (self.fc_m0_out_features - self.extra_m0_output_channels))
        
        x_0 = x_0.view(num_edges, -1, self.m_output_channels)
        #x.embedding[:, 0 : self.mappingReduced.m_size[0]] = x_0
        out.append(x_0)
        offset_rad = offset_rad + self.fc_m0.in_features
        offset = self.mappingReduced.m_size[0]

        # Compute the values for the m > 0 coefficients
        for m in range(1, max(self.mmax_list) + 1):
            # Get the m order coefficients
            x_m = x.embedding.narrow(1, offset, 2 * self.mappingReduced.m_size[m])
            x_m = x_m.reshape(num_edges, 2, -1)

            # Perform SO(2) convolution
            for idx, conv in enumerate(self.so2_m_conv):
                if idx == m - 1:  # Only process when we reach the target m
                    in_feat = conv.fc.in_features
                    if self.rad_func is not None:
                        x_edge_m = x_edge.narrow(1, offset_rad, in_feat)
                        x_edge_m = x_edge_m.reshape(num_edges, 1, in_feat)
                        x_m = x_m * x_edge_m
                    x_m = conv(x_m)
                    x_m = x_m.view(num_edges, -1, self.m_output_channels)
                    out.append(x_m)
                    offset = offset + 2 * self.mappingReduced.m_size[m]
                    offset_rad = offset_rad + conv.fc.in_features
                    # break

        out = torch.cat(out, dim=1)
        self.out_embedding.set_embedding(out)
        self.out_embedding.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # Reshape the spherical harmonics based on l (degree)
        self.out_embedding._l_primary(self.mappingReduced)

        return self.out_embedding, x_0_extra

class GateActivation(torch.nn.Module):
    def __init__(self, lmax, mmax, num_channels):
        super().__init__()

        self.lmax = lmax
        self.mmax = mmax
        self.num_channels = num_channels

        # compute `expand_index` based on `lmax` and `mmax`
        num_components = 0
        for l in range(1, self.lmax + 1):
            num_m_components = min((2 * l + 1), (2 * self.mmax + 1))
            num_components = num_components + num_m_components
        expand_index = torch.zeros([num_components]).long()
        start_idx = 0
        for l in range(1, self.lmax + 1):
            length = min((2 * l + 1), (2 * self.mmax + 1))
            expand_index[start_idx : (start_idx + length)] = (l - 1)
            start_idx = start_idx + length            
        self.register_buffer('expand_index', expand_index, persistent=False)

        self.scalar_act = torch.nn.SiLU() #SwiGLU(self.num_channels, self.num_channels)  # #
        self.gate_act   = torch.nn.Sigmoid() #torch.nn.SiLU() # #

    
    def forward(self, gating_scalars, input_tensors):
        '''
            `gating_scalars`: shape [N, lmax * num_channels]
            `input_tensors`: shape  [N, (lmax + 1) ** 2, num_channels]
        '''

        gating_scalars = self.gate_act(gating_scalars)
        gating_scalars = gating_scalars.reshape(gating_scalars.shape[0], self.lmax, self.num_channels)
        gating_scalars = torch.index_select(gating_scalars, dim=1, index=self.expand_index)

        input_tensors_scalars = input_tensors.narrow(1, 0, 1)
        input_tensors_scalars = self.scalar_act(input_tensors_scalars)

        input_tensors_vectors = input_tensors.narrow(1, 1, input_tensors.shape[1] - 1)
        input_tensors_vectors = input_tensors_vectors * gating_scalars

        output_tensors = torch.cat((input_tensors_scalars, input_tensors_vectors), dim=1)
        
        return output_tensors


class S2Activation(torch.nn.Module):
    '''
        Assume we only have one resolution
    '''
    def __init__(self, lmax, mmax):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.act = torch.nn.SiLU()

    def _get_grid_index(self, l: int, m: int, max_l: int) -> int:
        return l * (max_l + 1) + m

    def forward(self, inputs: torch.Tensor, SO3_grid: List[SO3_Grid]):
        # SO3_grid = list(SO3_grid)
        idx = self._get_grid_index(self.lmax, self.mmax, self.lmax) 
        to_grid_mat   = SO3_grid[idx].get_to_grid_mat(device=inputs.device)     # `device` is not used
        from_grid_mat = SO3_grid[idx].get_from_grid_mat(device=inputs.device)

        x_grid = torch.einsum("bai, zic -> zbac", to_grid_mat, inputs)
        x_grid = self.act(x_grid)
        outputs = torch.einsum("bai, zbac -> zic", from_grid_mat, x_grid)
        return outputs

    
class SeparableS2Activation(torch.nn.Module):
    def __init__(self, lmax, mmax):
        super().__init__()
        
        self.lmax = lmax
        self.mmax = mmax
        
        self.scalar_act = torch.nn.SiLU() 
        self.s2_act     = S2Activation(self.lmax, self.mmax)
        

    def forward(self, input_scalars: torch.Tensor, input_tensors: torch.Tensor, SO3_grid: List[SO3_Grid]):
        output_scalars = self.scalar_act(input_scalars)
        output_scalars = output_scalars.reshape(output_scalars.shape[0], 1, output_scalars.shape[-1])
        output_tensors = self.s2_act(input_tensors, SO3_grid)
        outputs = torch.cat(
            (output_scalars, output_tensors.narrow(1, 1, output_tensors.shape[1] - 1)), 
            dim=1
        )
        return outputs

class SO3_LinearV2(torch.nn.Module):
    def __init__(self, in_features, out_features, lmax, lmax_list, bias=True):
        '''
            1. Use `torch.einsum` to prevent slicing and concatenation
            2. Need to specify some behaviors in `no_weight_decay` and weight initialization.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax
        self.lmax_list = lmax_list
        self.weight = torch.nn.Parameter(torch.randn((self.lmax + 1), out_features, in_features))
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for l in range(lmax + 1):
            start_idx = l ** 2
            length = 2 * l + 1
            expand_index[start_idx : (start_idx + length)] = l
        self.register_buffer('expand_index', expand_index, persistent=False)

        self.register_buffer("dummy_buffer", torch.empty(0), persistent=False)
        self.out_embedding = SO3_Embedding(
            0, 
            self.lmax_list.copy(), 
            self.out_features, 
            device=self.dummy_buffer.device, 
            dtype=self.dummy_buffer.dtype
        )

    def forward(self, input_embedding: SO3_Embedding):

        weight = torch.index_select(self.weight, dim=0, index=self.expand_index) # [(L_max + 1) ** 2, C_out, C_in]
        out = torch.einsum('bmi, moi -> bmo', input_embedding.embedding, weight) # [N, (L_max + 1) ** 2, C_out]
        bias = self.bias.view(1, 1, self.out_features)
        out[:, 0:1, :] = out.narrow(1, 0, 1) + bias
        self.out_embedding.set_embedding(out)
        self.out_embedding.set_lmax_mmax(input_embedding.lmax_list.copy(), input_embedding.lmax_list.copy())

        return self.out_embedding
        

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, lmax={self.lmax})"


class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.alpha = negative_slope
        
    
    def forward(self, x):
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2
    
    
    def extra_repr(self):
        return 'negative_slope={}'.format(self.alpha)

class SO2EquivariantGraphAttention(torch.nn.Module):
    """
    SO2EquivariantGraphAttention: Perform MLP attention + non-linear message passing
        SO(2) Convolution with radial function -> S2 Activation -> SO(2) Convolution -> attention weights and non-linear messages
        attention weights * non-linear messages -> Linear

    Args:
        atom_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        output_channels (int):      Number of output channels
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        
        SO3_rotation (list:SO3_Rotation): Class to calculate Wigner-D matrices and rotate embeddings
        mappingReduced (CoefficientMappingModule): Class to convert l and m indices once node embedding is rotated
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        max_num_elements (int):     Maximum number of atomic numbers
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                        The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features
        use_m_share_rad (bool):     Whether all m components within a type-L vector of one channel share radial function weights

        activation (str):           Type of activation function
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
    """

    def __init__(
        self,
        atom_channels,
        hidden_channels,
        num_heads, 
        attn_alpha_channels,
        attn_value_channels, 
        output_channels,
        lmax_list,
        mmax_list,
        SO3_rotation, 
        mappingReduced, 
        SO3_grid, 
        max_num_elements,
        edge_channels_list,
        use_atom_edge_embedding=True, 
        use_m_share_rad=False,
        activation='scaled_silu', 
        use_s2_act_attn=False, 
        use_attn_renorm=True,
        use_gate_act=False, 
        use_sep_s2_act=True,
        alpha_drop=0.0,
    ):
        super(SO2EquivariantGraphAttention, self).__init__()
        
        self.atom_channels = atom_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.output_channels = output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(self.lmax_list)
        
        self.SO3_rotation = SO3_rotation
        self.mappingReduced = mappingReduced
        self.SO3_grid = SO3_grid
        
        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        self.max_num_elements = max_num_elements
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad

        if self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None
        
        self.use_s2_act_attn    = use_s2_act_attn
        self.use_attn_renorm    = use_attn_renorm
        self.use_gate_act       = use_gate_act
        self.use_sep_s2_act     = use_sep_s2_act
        
        assert not self.use_s2_act_attn     # since this is not used
        
        # Create SO(2) convolution blocks
        extra_m0_output_channels = None
        if not self.use_s2_act_attn:
            extra_m0_output_channels = self.num_heads * self.attn_alpha_channels
            if self.use_gate_act:
                extra_m0_output_channels = extra_m0_output_channels + max(self.lmax_list) * self.hidden_channels
            else:
                if self.use_sep_s2_act:
                    extra_m0_output_channels = extra_m0_output_channels + self.hidden_channels
        
        if self.use_m_share_rad:
            self.edge_channels_list = self.edge_channels_list + [2 * self.atom_channels * (max(self.lmax_list) + 1)]
            self.rad_func = RadialFunction(self.edge_channels_list)
            expand_index = torch.zeros([(max(self.lmax_list) + 1) ** 2]).long()
            for l in range(max(self.lmax_list) + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                expand_index[start_idx : (start_idx + length)] = l
            self.register_buffer('expand_index', expand_index, persistent=False)
        else:
            self.rad_func = None
            self.expand_index = None

        self.so2_conv_1 = SO2_Convolution(
            2 * self.atom_channels,
            self.hidden_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=(
                False if not self.use_m_share_rad 
                else True
            ),
            edge_channels_list=(
                self.edge_channels_list if not self.use_m_share_rad 
                else None
            ), 
            extra_m0_output_channels=extra_m0_output_channels # for attention weights and/or gate activation
        )

        if self.use_s2_act_attn:
            self.alpha_norm = None
            self.alpha_act = None
            self.alpha_dot = None
        else:
            if self.use_attn_renorm:
                self.alpha_norm = torch.nn.LayerNorm(self.attn_alpha_channels)
            else:
                self.alpha_norm = torch.nn.Identity()
            self.alpha_act = SmoothLeakyReLU()
            self.alpha_dot = torch.nn.Parameter(torch.randn(self.num_heads, self.attn_alpha_channels))
            std = 1.0 / math.sqrt(self.attn_alpha_channels)
            torch.nn.init.uniform_(self.alpha_dot, -std, std)
        
        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        if self.use_gate_act:
            self.gate_act = GateActivation(
                lmax=max(self.lmax_list), 
                mmax=max(self.mmax_list), 
                num_channels=self.hidden_channels
            )
        else:   
            if self.use_sep_s2_act:     
                # separable S2 activation
                self.s2_act = SeparableS2Activation(
                    lmax=max(self.lmax_list), 
                    mmax=max(self.mmax_list)
                )
            else:                       
                # S2 activation
                self.s2_act = S2Activation(
                    lmax=max(self.lmax_list), 
                    mmax=max(self.mmax_list)
                )
        
        self.so2_conv_2 = SO2_Convolution(
            self.hidden_channels,
            self.num_heads * self.attn_value_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None, 
            extra_m0_output_channels=(
                self.num_heads if self.use_s2_act_attn 
                else None
            ) # for attention weights
        )

        self.proj = SO3_LinearV2(self.num_heads * self.attn_value_channels, self.output_channels, lmax=self.lmax_list[0], lmax_list=self.lmax_list)

        self.register_buffer("dummy_buffer", torch.empty(0), persistent=False)
        self.clone = SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.atom_channels,
            self.dummy_buffer.device,
            self.dummy_buffer.dtype,
        )
        
        self.x_message = SO3_Embedding(
            0,
            self.lmax_list.copy(), 
            self.atom_channels * 2, 
            device=self.dummy_buffer.device, 
            dtype=self.dummy_buffer.dtype
        )
        
        
    def forward(
        self,
        x: SO3_Embedding,
        atomic_numbers: torch.Tensor,
        edge_distance: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rot_mat: torch.Tensor
    ) -> SO3_Embedding:
        
        # Compute edge scalar features (invariant to rotations)
        # Uses atomic numbers and edge distance as inputs
        if self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            x_edge = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)
        else:
            x_edge = edge_distance  

        # x_source = x.clone()
        # x_target = x.clone()
        x_source = self.clone
        x_source.set_embedding(x.embedding.clone())
        x_target = self.clone
        x_target.set_embedding(x.embedding.clone())

        x_source._expand_edge(edge_index[0])
        x_target._expand_edge(edge_index[1])
        
        x_message_data = torch.cat((x_source.embedding, x_target.embedding), dim=2)
        self.x_message.set_embedding(x_message_data)
        self.x_message.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # radial function (scale all m components within a type-L vector of one channel with the same weight)
        if self.use_m_share_rad and self.rad_func is not None:
            x_edge_weight = self.rad_func(x_edge)
            x_edge_weight = x_edge_weight.reshape(-1, (max(self.lmax_list) + 1), 2 * self.atom_channels)
            x_edge_weight = torch.index_select(x_edge_weight, dim=1, index=self.expand_index) # [E, (L_max + 1) ** 2, C]
            self.x_message.embedding = self.x_message.embedding * x_edge_weight

        # Rotate the irreps to align with the edge
        self.x_message._rotate(list(self.SO3_rotation), self.lmax_list, self.mmax_list, edge_rot_mat)

        # First SO(2)-convolution
        if self.use_s2_act_attn:
            self.x_message, x_0_extra = self.so2_conv_1(self.x_message, x_edge)
        else:
            self.x_message, x_0_extra = self.so2_conv_1(self.x_message, x_edge)
        
        # Activation
        x_alpha_num_channels = self.num_heads * self.attn_alpha_channels
        if x_0_extra is not None:
            if self.use_gate_act and hasattr(self, 'gate_act'):   
                # Gate activation
                x_0_gating = x_0_extra.narrow(1, x_alpha_num_channels, x_0_extra.shape[1] - x_alpha_num_channels) # for activation
                x_0_alpha  = x_0_extra.narrow(1, 0, x_alpha_num_channels) # for attention weights
                self.x_message.embedding = self.gate_act(x_0_gating, self.x_message.embedding)
            else:
                if self.use_sep_s2_act and hasattr(self, 's2_act'):
                    x_0_gating = x_0_extra.narrow(1, x_alpha_num_channels, x_0_extra.shape[1] - x_alpha_num_channels) # for activation
                    x_0_alpha  = x_0_extra.narrow(1, 0, x_alpha_num_channels) # for attention weights
                    self.x_message.embedding = self.s2_act(x_0_gating, self.x_message.embedding, list(self.SO3_grid))
                else:
                    x_0_alpha = x_0_extra
                    if isinstance(self.s2_act, SeparableS2Activation):
                        self.x_message.embedding = self.s2_act(x_0_alpha, self.x_message.embedding, list(self.SO3_grid))
                    else:
                        if isinstance(self.s2_act, S2Activation):
                            self.x_message.embedding = self.s2_act(self.x_message.embedding, list(self.SO3_grid))
                        else:
                            raise ValueError(f"Unknown S2 activation type: {type(self.s2_act)}")
        else:
            x_0_alpha = None
            alpha = torch.ones(len(edge_index[1]), device=self.x_message.embedding.device) # TODO: check if this is correct

        # Second SO(2)-convolution
        if self.use_s2_act_attn:
            self.x_message, x_0_extra = self.so2_conv_2(self.x_message, x_edge)
        else:
            self.x_message, _ = self.so2_conv_2(self.x_message, x_edge)
        
        # Attention weights
        if self.use_s2_act_attn:
            alpha = x_0_extra
        else:
            if x_0_alpha is not None:
                x_0_alpha = x_0_alpha.reshape(-1, self.num_heads, self.attn_alpha_channels)
                x_0_alpha = self.alpha_norm(x_0_alpha)
                x_0_alpha = self.alpha_act(x_0_alpha)
                alpha = torch.einsum('bik, ik -> bi', x_0_alpha, self.alpha_dot)
            else:
                alpha = torch.ones(len(edge_index[1]), device=self.x_message.embedding.device) # TODO: check if this is correct

        # Ensure alpha is a tensor before softmax
        if alpha is None:
            alpha = torch.ones(len(edge_index[1]), device=self.x_message.embedding.device) # TODO: check if this is correct

        alpha = torch_geometric.utils.softmax(alpha, edge_index[1])

        alpha = alpha.reshape(alpha.shape[0], 1, self.num_heads, 1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)
        
        # Attention weights * non-linear messages
        attn = self.x_message.embedding
        attn = attn.reshape(attn.shape[0], attn.shape[1], self.num_heads, self.attn_value_channels)
        attn = attn * alpha
        attn = attn.reshape(attn.shape[0], attn.shape[1], self.num_heads * self.attn_value_channels)
        self.x_message.embedding = attn

        # Rotate back the irreps
        self.x_message._rotate_inv(list(self.SO3_rotation), self.mappingReduced, edge_rot_mat)

        # Compute the sum of the incoming neighboring messages for each target node
        self.x_message._reduce_edge(edge_index[1], len(x.embedding))

        # Project
        out_embedding = self.proj(self.x_message)

        return out_embedding


class FeedForwardNetwork(torch.nn.Module):
    """
    FeedForwardNetwork: Perform feedforward network with S2 activation or gate activation

    Args:
        atom_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during feedforward network
        output_channels (int):      Number of output channels

        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        activation (str):           Type of activation function
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs. 
        use_sep_s2_act (bool):      If `True`, use separable grid MLP when `use_grid_mlp` is True.
    """

    def __init__(
        self,
        atom_channels,
        hidden_channels, 
        output_channels,
        lmax_list,
        mmax_list,
        SO3_grid,  
        activation='scaled_silu', 
        use_gate_act=False, 
        use_grid_mlp=False, 
        use_sep_s2_act=True
    ):
        super(FeedForwardNetwork, self).__init__()
        self.atom_channels = atom_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)
        self.atom_channels_all = self.num_resolutions * self.atom_channels
        self.SO3_grid = SO3_grid
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.max_lmax = max(self.lmax_list)
        
        if isinstance(self.hidden_channels, torch.Tensor):
            self.hidden_channels = self.hidden_channels.item()
        assert isinstance(self.hidden_channels, int)
        if isinstance(self.max_lmax, torch.Tensor):
            self.max_lmax = self.max_lmax.item()
        assert isinstance(self.max_lmax, int)
        self.so3_linear_1 = SO3_LinearV2(self.atom_channels_all, self.hidden_channels, lmax=self.max_lmax, lmax_list=self.lmax_list)
        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                self.scalar_mlp = nn.Sequential(
                    nn.Linear(self.atom_channels_all, self.hidden_channels, bias=True), 
                    nn.SiLU(), 
                )
            else:
                self.scalar_mlp = None
            self.grid_mlp = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False), 
                nn.SiLU(), 
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
                nn.SiLU(), 
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
            )
        else:
            if self.use_gate_act:
                self.gating_linear = torch.nn.Linear(self.atom_channels_all, self.max_lmax * self.hidden_channels)
                self.gate_act = GateActivation(self.max_lmax, self.max_lmax, self.hidden_channels)
            else:
                if self.use_sep_s2_act:
                    self.gating_linear = torch.nn.Linear(self.atom_channels_all, self.hidden_channels)
                    self.s2_act = SeparableS2Activation(self.max_lmax, self.max_lmax)
                else:
                    self.gating_linear = None
                    self.s2_act = S2Activation(self.max_lmax, self.max_lmax)
        self.so3_linear_2 = SO3_LinearV2(self.hidden_channels, self.output_channels, lmax=self.max_lmax, lmax_list=self.lmax_list)
        
    
    def forward(self, input_embedding: SO3_Embedding):
        gating_scalars: Optional[torch.Tensor] = None
        if self.use_grid_mlp:
            if self.use_sep_s2_act and hasattr(self, 'scalar_mlp'):
                gating_scalars = self.scalar_mlp(input_embedding.embedding.narrow(1, 0, 1))    
        else:
            if self.gating_linear is not None and hasattr(self, 'gating_linear'):
                gating_scalars = self.gating_linear(input_embedding.embedding.narrow(1, 0, 1))

        input_embedding = self.so3_linear_1(input_embedding)
        
        if self.use_grid_mlp and hasattr(self, 'grid_mlp'):
            # Project to grid
            input_embedding_grid = input_embedding.to_grid(list(self.SO3_grid), lmax=self.max_lmax)
            # Perform point-wise operations
            input_embedding_grid = self.grid_mlp(input_embedding_grid)
            # Project back to spherical harmonic coefficients
            input_embedding._from_grid(input_embedding_grid, list(self.SO3_grid), lmax=self.max_lmax)

            if self.use_sep_s2_act:
                input_embedding.embedding = torch.cat(
                    (gating_scalars, input_embedding.embedding.narrow(1, 1, input_embedding.embedding.shape[1] - 1)), 
                    dim=1
                )
        else:
            if self.use_gate_act and hasattr(self, 'gate_act'):
                input_embedding.embedding = self.gate_act(gating_scalars, input_embedding.embedding)
            else:
                if self.use_sep_s2_act and hasattr(self, 's2_act'):
                    if isinstance(self.s2_act, SeparableS2Activation) and isinstance(gating_scalars, torch.Tensor):
                        input_embedding.embedding = self.s2_act(gating_scalars, input_embedding.embedding, list(self.SO3_grid))
                    elif isinstance(self.s2_act, S2Activation):
                        input_embedding.embedding = self.s2_act(input_embedding.embedding, list(self.SO3_grid))
                    else:
                        raise ValueError("Unknown S2 activation type")

        input_embedding = self.so3_linear_2(input_embedding)

        return input_embedding

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class GraphDropPath(nn.Module):
    '''
        Consider batch for graph data when dropping paths.
    '''
    def __init__(self, drop_prob=None):
        super(GraphDropPath, self).__init__()
        self.drop_prob = drop_prob
        

    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        # Convert batch_size tensor to integer
        batch_size = int(batch.max().item()) + 1
        shape = (batch_size,) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        ones = torch.ones(shape, dtype=x.dtype, device=x.device)
        drop = drop_path(ones, self.drop_prob, self.training)
        out = x * drop[batch]
        return out
    
    
    def extra_repr(self):
        return 'drop_prob={}'.format(self.drop_prob)

class EquivariantDropoutArraySphericalHarmonics(nn.Module):
    def __init__(self, drop_prob, drop_graph=False):
        super(EquivariantDropoutArraySphericalHarmonics, self).__init__()
        self.drop_prob = drop_prob
        self.drop = torch.nn.Dropout(drop_prob, True)
        self.drop_graph = drop_graph
        
        
    def forward(self, x, batch=None):
        if not self.training or self.drop_prob == 0.0:
            return x
        assert len(x.shape) == 3

        if self.drop_graph:
            assert batch is not None
            # batch_size = batch
            batch_size = batch.max() + 1
            shape = (batch_size, 1, x.shape[2])
            mask = torch.ones(shape, dtype=x.dtype, device=x.device)
            mask = self.drop(mask)
            out = x * mask[batch]
        else:
            shape = (x.shape[0], 1, x.shape[2])
            mask = torch.ones(shape, dtype=x.dtype, device=x.device)
            mask = self.drop(mask)
            out = x * mask

        return out
    
    
    def extra_repr(self):
        return 'drop_prob={}, drop_graph={}'.format(self.drop_prob, self.drop_graph)

class TransBlockV2(torch.nn.Module):
    """

    Args:
        atom_channels (int):      Number of spherical channels
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        output_channels (int):      Number of output channels

        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        
        SO3_rotation (list:SO3_Rotation): Class to calculate Wigner-D matrices and rotate embeddings
        mappingReduced (CoefficientMappingModule): Class to convert l and m indices once node embedding is rotated
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        max_num_elements (int):     Maximum number of atomic numbers
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                        The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features
        use_m_share_rad (bool):     Whether all m components within a type-L vector of one channel share radial function weights

        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFN.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh'])

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN
    """

    def __init__(
        self,
        atom_channels,
        attn_hidden_channels,
        num_heads,
        attn_alpha_channels, 
        attn_value_channels,
        ffn_hidden_channels,
        output_channels, 

        lmax_list,
        mmax_list,
        
        SO3_rotation,
        mappingReduced,
        SO3_grid,

        max_num_elements,
        edge_channels_list,
        use_atom_edge_embedding=True,
        use_m_share_rad=False,

        attn_activation='silu',
        use_s2_act_attn=False, 
        use_attn_renorm=True,
        ffn_activation='silu',
        use_gate_act=False, 
        use_grid_mlp=False,
        use_sep_s2_act=True,

        norm_type='rms_norm_sh',

        alpha_drop=0.0, 
        drop_path_rate=0.0, 
        proj_drop=0.0
    ):
        super(TransBlockV2, self).__init__()

        max_lmax = max(lmax_list)
        self.norm_1 = get_normalization_layer(norm_type, lmax=max_lmax, num_channels=atom_channels)

        self.ga = SO2EquivariantGraphAttention(
            atom_channels=atom_channels,
            hidden_channels=attn_hidden_channels,
            num_heads=num_heads, 
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels, 
            output_channels=atom_channels,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            SO3_rotation=SO3_rotation, 
            mappingReduced=mappingReduced, 
            SO3_grid=SO3_grid, 
            max_num_elements=max_num_elements,
            edge_channels_list=edge_channels_list,
            use_atom_edge_embedding=use_atom_edge_embedding, 
            use_m_share_rad=use_m_share_rad,
            activation=attn_activation, 
            use_s2_act_attn=use_s2_act_attn,
            use_attn_renorm=use_attn_renorm,
            use_gate_act=use_gate_act,
            use_sep_s2_act=use_sep_s2_act,
            alpha_drop=alpha_drop,
        )

        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0. else None
        self.proj_drop = EquivariantDropoutArraySphericalHarmonics(proj_drop, drop_graph=False) if proj_drop > 0.0 else None

        self.norm_2 = get_normalization_layer(norm_type, lmax=max_lmax, num_channels=atom_channels)
        
        self.ffn = FeedForwardNetwork(
            atom_channels=atom_channels,
            hidden_channels=ffn_hidden_channels, 
            output_channels=output_channels,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            SO3_grid=SO3_grid,  
            activation=ffn_activation,
            use_gate_act=use_gate_act,
            use_grid_mlp=use_grid_mlp,
            use_sep_s2_act=use_sep_s2_act
        )

        if atom_channels != output_channels:
            self.ffn_shortcut = SO3_LinearV2(atom_channels, output_channels, lmax=max_lmax, lmax_list=lmax_list)
        else:
            self.ffn_shortcut = None
    
    def forward(
        self,
        x: SO3_Embedding,              # SO3_Embedding
        atomic_numbers: torch.Tensor,
        edge_distance: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rot_mat: torch.Tensor,
        batch: torch.Tensor           # for GraphDropPath
    ):
        output_embedding = x
        
        x_res = output_embedding.embedding
        output_embedding.embedding = self.norm_1(output_embedding.embedding)
        output_embedding = self.ga(output_embedding, 
            atomic_numbers,
            edge_distance,
            edge_index,
            edge_rot_mat)
        
        if self.drop_path is not None:
            output_embedding.embedding = self.drop_path(output_embedding.embedding, batch)
        if self.proj_drop is not None:
            output_embedding.embedding = self.proj_drop(output_embedding.embedding, batch)

        output_embedding.embedding = output_embedding.embedding + x_res

        x_res = output_embedding.embedding
        output_embedding.embedding = self.norm_2(output_embedding.embedding)
        output_embedding = self.ffn(output_embedding)

        if self.drop_path is not None:
            output_embedding.embedding = self.drop_path(output_embedding.embedding, batch)
        if self.proj_drop is not None:
            output_embedding.embedding = self.proj_drop(output_embedding.embedding, batch)

        if self.ffn_shortcut is not None:
            shortcut_embedding = SO3_Embedding(
                0, 
                output_embedding.lmax_list.copy(), 
                self.ffn_shortcut.in_features, 
                device=output_embedding.device, 
                dtype=output_embedding.dtype
            )
            shortcut_embedding.set_embedding(x_res)
            shortcut_embedding.set_lmax_mmax(output_embedding.lmax_list.copy(), output_embedding.lmax_list.copy())
            shortcut_embedding = self.ffn_shortcut(shortcut_embedding)
            x_res = shortcut_embedding.embedding

        output_embedding.embedding = output_embedding.embedding + x_res

        return output_embedding


class ModuleListInfo(torch.nn.ModuleList):
    def __init__(self, info_str, modules=None):
        super().__init__(modules)
        self.info_str = str(info_str)

    def __repr__(self): 
        return self.info_str 
    

class EquiformerV2(nn.Module):
    def __init__(self, cutoff: float, device='cpu', num_features='128',num_interactions=3, compute_forces=False, **kwargs):
        super().__init__()
        # Initialize the basic parameters
        self._AVG_NUM_NODES  = 1 #77.81317
        self._AVG_DEGREE     = 1 #23.395238876342773    # IS2RE: 100k, max_radius = 5, max_neighbors = 100
        self.lmax_list = kwargs.get('lmax_list', [4]) # [6]
        self.mmax_list = kwargs.get('mmax_list', [2])
        self.grid_resolution = kwargs.get('grid_resolution', None) #Initialize the transformations between spherical and grid representations
        # print('self.grid_resolution:', self.grid_resolution)

        self.device = torch.device(device)
        self.dtype = torch.float32 
        self.compute_forces = compute_forces
        self.cutoff = cutoff
        self.num_resolutions = len(self.lmax_list)

        self.atom_channels=num_features
        self.max_num_elements = 119
        self.atom_channels_all = self.num_resolutions * self.atom_channels
        self.atom_embedding = nn.Embedding(self.max_num_elements, self.atom_channels_all)

        # Initialize the blocks for each layer of EquiformerV2
        self.num_layers = num_interactions # 12
        self.attn_hidden_channels = num_features
        self.num_heads = 8
        self.attn_alpha_channels = num_features
        self.attn_value_channels = 16
        self.ffn_hidden_channels = num_features*2 #4
        self.use_m_share_rad = False
        self.distance_function = 'gaussian'
        self.num_distance_basis = num_features*2 #4
        self.attn_activation = 'scaled_silu'
        self.use_s2_act_attn = False
        self.use_attn_renorm = True
        self.ffn_activation = 'scaled_silu'
        self.use_gate_act = False
        self.use_grid_mlp = True # False
        self.use_sep_s2_act = True
        self.alpha_drop = 0.05 #0.1
        self.drop_path_rate = 0.02 #0.05
        self.proj_drop = 0.0
        self.norm_type = 'rms_norm_sh'

        self.weight_init = 'normal'
        assert self.weight_init in ['normal', 'uniform']

        self.distance_function = 'gaussian'
        assert self.distance_function in [
            'gaussian',
        ]
        if self.distance_function == 'gaussian':
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                300, #600,
                2.0,
            )
        else:
            raise ValueError
        
        self.share_atom_edge_embedding = False
        self.use_atom_edge_embedding = True
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding

        self.edge_channels = num_features
        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [self.edge_channels] * 2

        # Initialize atom edge embedding
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]

        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i], self.device))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list, self.device)

        max_l = max(self.lmax_list)
        self.SO3_grid = nn.ModuleList()
        for l in range(max_l + 1):
            for m in range(max_l + 1):
                self.SO3_grid.append(
                    SO3_Grid(
                        l, 
                        m, 
                        resolution=self.grid_resolution, 
                        normalization='component',
                        device=self.device,
                    )
                )

        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.atom_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=self._AVG_DEGREE
        )
        
        # Seperable layer norm, and equivariant graph attention
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV2(
                self.atom_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.atom_channels, 
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop, 
                self.drop_path_rate,
                self.proj_drop
            )
            self.blocks.append(block)

        # Output blocks for energy and forces
        self.norm = get_normalization_layer(self.norm_type, lmax=max(self.lmax_list), num_channels=self.atom_channels)
        self.energy_block = FeedForwardNetwork(
            self.atom_channels,
            self.ffn_hidden_channels, 
            1,
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,  
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act
        )
        if self.compute_forces:
            self.force_block = SO2EquivariantGraphAttention(
                self.atom_channels,
                self.attn_hidden_channels,
                self.num_heads, 
                self.attn_alpha_channels,
                self.attn_value_channels, 
                1,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation, 
                self.mappingReduced, 
                self.SO3_grid, 
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding, 
                self.use_m_share_rad,
                self.attn_activation, 
                self.use_s2_act_attn, 
                self.use_attn_renorm,
                self.use_gate_act,
                self.use_sep_s2_act,
                alpha_drop=0.0
            )
            
        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

        # Initialize SO3_Embedding
        self.x = SO3_Embedding(
            0,  # Will be set in forward pass
            self.lmax_list,
            self.atom_channels,
            self.device,
            self.dtype,
        )

    def _get_grid_index(self, l: int, m: int, max_l: int) -> int:
        return l * (max_l + 1) + m


    def forward(self, data: AtomsData):
        """
        Args:
            data (AtomsData): A NamedTuple of model inputs

        Returns:
            dict: A dictionary with keys 'energy' and 'forces'
        """

        atomic_numbers = data.atomic_numbers.long()
        edge_index = data.edge_indices.T
        edge_vectors = data.edge_vectors
        positions = data.positions
        image_indices = data.image_indices
        edge_dist = torch.linalg.norm(edge_vectors, dim=1)
        num_atoms = len(atomic_numbers)

        ###############################################################
        # Embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        self.x.set_embedding(torch.zeros(
            num_atoms,
            self.x.num_coefficients,
            self.atom_channels,
            device=self.device,
            dtype=positions.dtype,
        ))
        self.x.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        # Atom embedding
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                self.x.embedding[:, offset_res, :] = self.atom_embedding(atomic_numbers)
            else:
                self.x.embedding[:, offset_res, :] = self.atom_embedding(
                    atomic_numbers
                    )[:, offset : offset + self.atom_channels]
            offset = offset + self.atom_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        edge_dist = self.distance_expansion(edge_dist)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding and hasattr(self, 'source_embedding') and hasattr(self, 'target_embedding'):
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_dist = torch.cat((edge_dist, source_embedding, target_embedding), dim=1)

        # Edge-degree embedding
        edge_rot_mat = init_edge_rot_mat(edge_vectors) # Compute 3x3 rotation matrix per edge
        edge_degree = self.edge_degree_embedding( # torchscript error
            atomic_numbers,
            edge_dist,
            edge_index,
            edge_rot_mat)
        self.x.embedding = self.x.embedding + edge_degree.embedding

        ###############################################################
        # Seperable layer norm, and equivariant graph attention
        ###############################################################
        if image_indices is None:
            image_indices = torch.zeros_like(atomic_numbers, dtype=torch.long)
        assert image_indices is not None
        for _, block in enumerate(self.blocks):
            self.x = block(
                self.x,
                atomic_numbers,
                edge_dist,
                edge_index,
                edge_rot_mat,
                batch=image_indices # data.batch    # for GraphDropPath
            )

        # Final layer norm
        self.x.embedding = self.norm(self.x.embedding)

        ###############################################################
        # Energy estimation
        ###############################################################
        node_energy = self.energy_block(self.x) # feedforward NN
        node_energy = node_energy.embedding.narrow(1, 0, 1)
        energy = torch.zeros(len(data.num_atoms), device=node_energy.device, dtype=node_energy.dtype)
        energy.index_add_(0, image_indices, node_energy.view(-1))
        # energy = energy / self._AVG_NUM_NODES
        data = replace_properties(data, energy=energy)
        
        atomic_energy = node_energy.view(-1)
        data = replace_properties(data, atomic_energy=atomic_energy)
        ###############################################################
        # Force estimation
        ###############################################################
        if self.compute_forces:
            forces = self.force_block(self.x,
                atomic_numbers,
                edge_dist,
                edge_index,
                edge_rot_mat)
            forces = forces.embedding.narrow(1, 1, 3)
            forces = forces.view(-1, 3)
            data = replace_properties(data, forces=forces)

        return data

    
    def _init_weights(self, m):
        if (isinstance(m, torch.nn.Linear)
            or isinstance(m, SO3_LinearV2)
        ):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == 'normal':
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    
    def _uniform_init_rad_func_linear_weights(self, m):
        if (isinstance(m, RadialFunction)):
            m.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)