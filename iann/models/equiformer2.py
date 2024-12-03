import torch
from torch import nn


class SO3_Embedding():
    """
    Helper functions for performing operations on irreps embedding

    Args:
        length (int):           Batch size
        lmax_list (list:int):   List of maximum degree of the spherical harmonics
        num_channels (int):     Number of channels
        device:                 Device of the output
        dtype:                  type of the output tensors
    """

    def __init__(
        self,
        length,
        lmax_list,
        num_channels,
        device,
        dtype,
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

        embedding = torch.zeros(
            length,
            self.num_coefficients,
            self.num_channels,
            device=self.device,
            dtype=self.dtype,
        )

        self.set_embedding(embedding)
        self.set_lmax_mmax(lmax_list, lmax_list.copy())


    # Clone an embedding of irreps
    def clone(self):
        clone = SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.num_channels,
            self.device,
            self.dtype,
        )
        clone.set_embedding(self.embedding.clone())
        return clone


    # Initialize an embedding of irreps
    def set_embedding(self, embedding):
        self.length = len(embedding)
        self.embedding = embedding


    # Set the maximum order to be the maximum degree
    def set_lmax_mmax(self, lmax_list, mmax_list):
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list


    # Expand the node embeddings to the number of edges
    def _expand_edge(self, edge_idx):
        embedding = self.embedding[edge_idx]
        self.set_embedding(embedding)


    # Initialize an embedding of irreps of a neighborhood
    def expand_edge(self, edge_idx):
        x_expand = SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.num_channels,
            self.device,
            self.dtype,
        )
        x_expand.set_embedding(self.embedding[edge_idx])
        return x_expand


    # Compute the sum of the embeddings of the neighborhood
    def _reduce_edge(self, edge_idx, num_nodes):
        new_embedding = torch.zeros(
            num_nodes,
            self.num_coefficients,
            self.num_channels,
            device=self.embedding.device,
            dtype=self.embedding.dtype,
        )
        new_embedding.index_add_(0, edge_idx, self.embedding)
        self.set_embedding(new_embedding)


    # Reshape the embedding l -> m
    def _m_primary(self, mapping):
        self.embedding = torch.einsum("nac, ba -> nbc", self.embedding, mapping.to_m)


    # Reshape the embedding m -> l
    def _l_primary(self, mapping):
        self.embedding = torch.einsum("nac, ab -> nbc", self.embedding, mapping.to_m)


    # Rotate the embedding
    def _rotate(self, SO3_rotation, lmax_list, mmax_list):
        
        if self.num_resolutions == 1:
            embedding_rotate = SO3_rotation[0].rotate(self.embedding, lmax_list[0], mmax_list[0])
        else:
            offset = 0
            embedding_rotate = torch.tensor([], device=self.device, dtype=self.dtype)
            for i in range(self.num_resolutions):
                num_coefficients = int((self.lmax_list[i] + 1) ** 2)
                embedding_i = self.embedding[:, offset : offset + num_coefficients]
                embedding_rotate = torch.cat([
                        embedding_rotate,
                        SO3_rotation[i].rotate(embedding_i, lmax_list[i], mmax_list[i])],
                    dim=1)
                offset = offset + num_coefficients

        self.embedding = embedding_rotate
        self.set_lmax_mmax(lmax_list.copy(), mmax_list.copy())


    # Rotate the embedding by the inverse of the rotation matrix
    def _rotate_inv(self, SO3_rotation, mappingReduced):

        if self.num_resolutions == 1:
            embedding_rotate = SO3_rotation[0].rotate_inv(self.embedding, self.lmax_list[0], self.mmax_list[0])
        else:
            offset = 0
            embedding_rotate = torch.tensor([], device=self.device, dtype=self.dtype)
            for i in range(self.num_resolutions):
                num_coefficients = mappingReduced.res_size[i]
                embedding_i = self.embedding[:, offset : offset + num_coefficients]
                embedding_rotate = torch.cat([
                        embedding_rotate,
                        SO3_rotation[i].rotate_inv(embedding_i, self.lmax_list[i], self.mmax_list[i])],
                    dim=1)
                offset = offset + num_coefficients
        self.embedding = embedding_rotate

        # Assume mmax = lmax when rotating back
        for i in range(self.num_resolutions):
            self.mmax_list[i] = int(self.lmax_list[i])
        self.set_lmax_mmax(self.lmax_list, self.mmax_list)


    # Compute point-wise spherical non-linearity
    def _grid_act(self, SO3_grid, act, mappingReduced):
        offset = 0
        for i in range(self.num_resolutions):

            num_coefficients = mappingReduced.res_size[i]

            if self.num_resolutions == 1:
                x_res = self.embedding
            else:
                x_res = self.embedding[:, offset : offset + num_coefficients].contiguous()
            to_grid_mat   = SO3_grid[self.lmax_list[i]][self.mmax_list[i]].get_to_grid_mat(self.device)
            from_grid_mat = SO3_grid[self.lmax_list[i]][self.mmax_list[i]].get_from_grid_mat(self.device)

            x_grid = torch.einsum("bai, zic -> zbac", to_grid_mat, x_res)
            x_grid = act(x_grid)
            x_res = torch.einsum("bai, zbac -> zic", from_grid_mat, x_grid)
            if self.num_resolutions == 1:
                self.embedding = x_res
            else:
               self.embedding[:, offset : offset + num_coefficients] = x_res
            offset = offset + num_coefficients


    # Compute a sample of the grid
    def to_grid(self, SO3_grid, lmax=-1):
        if lmax == -1:
            lmax = max(self.lmax_list)

        to_grid_mat_lmax = SO3_grid[lmax][lmax].get_to_grid_mat(self.device)
        grid_mapping     = SO3_grid[lmax][lmax].mapping

        offset = 0
        x_grid = torch.tensor([], device=self.device)

        for i in range(self.num_resolutions):
            num_coefficients = int((self.lmax_list[i] + 1) ** 2)
            if self.num_resolutions == 1:
                x_res = self.embedding
            else:
                x_res = self.embedding[:, offset : offset + num_coefficients].contiguous()
            to_grid_mat = to_grid_mat_lmax[:, :, grid_mapping.coefficient_idx(self.lmax_list[i], self.lmax_list[i])]
            x_grid = torch.cat([x_grid, torch.einsum("bai, zic -> zbac", to_grid_mat, x_res)], dim=3)
            offset = offset + num_coefficients

        return x_grid


    # Compute irreps from grid representation
    def _from_grid(self, x_grid, SO3_grid, lmax=-1):
        if lmax == -1:
            lmax = max(self.lmax_list)

        from_grid_mat_lmax = SO3_grid[lmax][lmax].get_from_grid_mat(self.device)
        grid_mapping       = SO3_grid[lmax][lmax].mapping

        offset = 0
        offset_channel = 0
        for i in range(self.num_resolutions):
            from_grid_mat = from_grid_mat_lmax[:, :, grid_mapping.coefficient_idx(self.lmax_list[i], self.lmax_list[i])]
            if self.num_resolutions == 1:
                temp = x_grid
            else:
                temp = x_grid[:, :, :, offset_channel : offset_channel + self.num_channels]
            x_res = torch.einsum("bai, zbac -> zic", from_grid_mat, temp)
            num_coefficients = int((self.lmax_list[i] + 1) ** 2)
            
            if self.num_resolutions == 1:
                self.embedding = x_res
            else: 
                self.embedding[:, offset : offset + num_coefficients] = x_res
            
            offset = offset + num_coefficients
            offset_channel = offset_channel + self.num_channels



class EquiformerV2(nn.Module()):
    def __init__(self, node_size: int, edge_size: int, cutoff: float):
        super().__init__()
        
        self.edge_size = edge_size
        self.node_size = node_size
        self.cutoff = cutoff

    def forward(self, data):
        atomic_numbers = data['elems']
        edge_diff = input_dict['n_diff']
        edge_dist = torch.linalg.norm(edge_diff, dim=1)
        edge_idx = data["pairs"]

        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device

        # atomic_numbers = data.atomic_numbers.long()
        num_atoms = len(atomic_numbers)
        # pos = data.pos

        # (
        #     edge_idx,
        #     edge_dist,
        #     edge_diff,
        #     cell_offsets,
        #     _,  # cell offset distances
        #     neighbors,
        # ) = self.generate_graph(data)

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, edge_idx, edge_diff
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(
                    atomic_numbers
                    )[:, offset : offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        edge_dist = self.distance_expansion(edge_dist)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_idx[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_idx[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_dist = torch.cat((edge_dist, source_embedding, target_embedding), dim=1)

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            atomic_numbers,
            edge_dist,
            edge_idx)
        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            x = self.blocks[i](
                x,                  # SO3_Embedding
                atomic_numbers,
                edge_dist,
                edge_idx,
                batch=data.batch    # for GraphDropPath
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        ###############################################################
        # Energy estimation
        ###############################################################
        node_energy = self.energy_block(x) 
        node_energy = node_energy.embedding.narrow(1, 0, 1)
        energy = torch.zeros(len(data.natoms), device=node_energy.device, dtype=node_energy.dtype)
        energy.index_add_(0, data.batch, node_energy.view(-1))
        energy = energy / _AVG_NUM_NODES

        ###############################################################
        # Force estimation
        ###############################################################
        if self.regress_forces:
            forces = self.force_block(x,
                atomic_numbers,
                edge_dist,
                edge_idx)
            forces = forces.embedding.narrow(1, 1, 3)
            forces = forces.view(-1, 3)            
            
        if not self.regress_forces:
            return energy
        else:
            return energy, forces