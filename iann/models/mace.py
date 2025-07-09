import torch
from torch import nn
from e3nn import o3
from e3nn.nn import Activation
from e3nn.util.jit import compile_mode
import abc,warnings
from ase.data import atomic_numbers
from e3nn.o3 import Linear
import math
from typing import Optional, Callable, Tuple
from typing import List, Dict, Union
from e3nn.nn import FullyConnectedNet
from e3nn.util.codegen import CodeGenMixin
import opt_einsum_fx, collections
from iann.data.data import AtomsData, replace_properties

activation_fn = {
    "silu": torch.nn.SiLU(),
    "tanh": torch.tanh,
    "abs": torch.abs,
    "None": None,
}

class Transform(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__()
    
    @abc.abstractmethod
    def forward(self):
        raise NotImplementedError

class TypeMapper(Transform):
    def __init__(
        self,
        species: Optional[List[str]]=None,
        symbol_to_type: Optional[Dict[str, int]]=None,
    ) -> None:
        super().__init__()
        if species is not None:
            if symbol_to_type is not None:
                raise TypeError("Cannot give both `species` and `symbol_to_type`")
            numbers = [atomic_numbers[s] for s in species]
            # sort chemical species
            species = [e[1] for e in sorted(zip(numbers, species))]
            symbol_to_type = {k: idx for idx, k in enumerate(species)}
        self.symbol_to_type = symbol_to_type
        
        if self.symbol_to_type is not None:
            for sym, type in self.symbol_to_type.items():
                assert sym in atomic_numbers, f"Invalid chemical symbol {sym}"
                assert 0 <= type, f"Invalid type number {type}"
            # 119 elements
            Z_to_index = torch.full(size=(119,), fill_value=-1, dtype=torch.long)
            for sym, type in self.symbol_to_type.items():
                Z_to_index[atomic_numbers[sym]] = type
            index_to_Z = torch.zeros(size=(len(self.symbol_to_type),), dtype=torch.long)
            for sym, type in self.symbol_to_type.items():
                index_to_Z[type] = atomic_numbers[sym]
        
            self.register_buffer("Z_to_index", Z_to_index)
            self.register_buffer("index_to_Z", index_to_Z)
        else:
            raise ValueError("`species` or `symbol_to_type` should be given!")
        
    def forward(self, data):
        if hasattr(data, 'atomic_types'):
            warnings.warn("Data already contains mapped types. This will be overwrited.")
        
        data.atomic_types = self.transform(data.atomic_numbers)
        assert torch.all(data.atomic_types >= 0), "Provided data contains species not defined in TypeMapper!"
        return data
        
    def transform(self, numbers: torch.Tensor) -> torch.Tensor:
        if numbers.max() > 119 or numbers.min() < 1:
            raise ValueError("Provided atomic numbers are not in the periodic table!")
        types = self.Z_to_index[numbers]
        return types
    
    def untransform(self, types: torch.Tensor) -> torch.Tensor:
        return self.index_to_Z[types]

#@compile_mode("script")
class OneHotAtomEncoding(torch.nn.Module):
    """Copmute a one-hot floating point encoding of atoms' discrete atom types.

    Args:
        set_features: If ``True`` (default), ``node_features`` will be set in addition to ``node_attrs``.
    """

    num_elements: int

    def __init__(
        self,
        num_elements: Optional[int] = None,
        species: Optional[List[str]] = None,
        set_features: bool=True,
    ):
        super().__init__()
        self.num_elements = num_elements
        self.set_features = set_features
        self.species = species
        
        if self.species is not None:
            self.type_mapper = TypeMapper(self.species)
            self.num_elements = len(self.species)
        else:
            self.num_elements = 119
            self.type_mapper = None
        # output node feature irreps
        self.irreps_out = {
            'node_attr': o3.Irreps([(self.num_elements, (0, 1))])
        }
        if self.set_features:
            self.irreps_out['node_feat'] = self.irreps_out['node_attr']
            
    def forward(self, data: AtomsData):
        if self.type_mapper is not None:
            atomic_types = self.type_mapper(data)
        else:
            atomic_types = data.atomic_numbers - 1
    
            # data = data._replace(atomic_types=data.atomic_numbers - 1)
            # data = replace_properties(data, atomic_types=data.atomic_numbers - 1)
        onehot = torch.nn.functional.one_hot(
            atomic_types, num_classes=self.num_elements
        ).to(device=data.positions.device, dtype=data.positions.dtype)
        
        data = replace_properties(data, node_attr=onehot)

        if self.set_features:
            data = replace_properties(data, node_feat=onehot)
        return data
    
class AtomwiseLinear(torch.nn.Module):
    def __init__(
        self,
        irreps_in: Optional[o3.Irreps]=None,
        irreps_out: Optional[o3.Irreps]=None,
    ):
        super().__init__()
        self.irreps_in: Optional[o3.Irreps] = irreps_in
        if irreps_out is None:
            irreps_out = irreps_in
        self.irreps_out = irreps_out
        
        self.linear = Linear(
            irreps_in=self.irreps_in, irreps_out=self.irreps_out
        )

    def forward(self, data: AtomsData):
        if data.node_feat is None:
            raise RuntimeError("node_feat must not be None")
        node_feat = data.node_feat
        assert isinstance(node_feat, torch.Tensor)
        node_feat = self.linear(node_feat)
        data = replace_properties(data, node_feat=node_feat)
        return data

#@compile_mode("script") 
class AtomwiseNonLinear(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        irreps_out: o3.Irreps=o3.Irreps("1x0e"),
    ):
        super().__init__()
        self.MLP_irreps = MLP_irreps
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.MLP_irreps)
        self.non_linearity = Activation(irreps_in=self.MLP_irreps, acts=[gate])
        self.linear_2 = o3.Linear(
            irreps_in=self.MLP_irreps, irreps_out=irreps_out
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]

class RadialBasis(torch.nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self):
        pass

class BesselBasis(RadialBasis):
    cutoff: float
    prefactor: float

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

#@torch.jit.script
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
    _factor: float
    p: float

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
    
#@compile_mode("script")
class RadialBasisEdgeEncoding(torch.nn.Module):
    out_field: str

    def __init__(
        self,
        basis: RadialBasis,
        cutoff_fn: CutoffFunction,
    ):
        super().__init__()
        self.basis = basis
        self.cutoff_fn = cutoff_fn
        
        # output edge dist irreps
        self.irreps_out = self.basis.irreps_out

    def forward(self, data: AtomsData):
        edge_vectors = data.edge_vectors
        edge_dist = torch.linalg.norm(edge_vectors, dim=1)
        edge_dist_embedding = (
            self.basis(edge_dist) * self.cutoff_fn(edge_dist)[:, None]
        )
        data = replace_properties(data, edge_dist_embedding=edge_dist_embedding)

        return data

class SphericalHarmonicEdgeAttrs(torch.nn.Module):
    def __init__(
        self,
        edge_sh_irreps: o3.Irreps,
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
    ):
        super().__init__()
        
        self.edge_sh_irreps = edge_sh_irreps
        self.sh = o3.SphericalHarmonics(
            self.edge_sh_irreps, edge_sh_normalize, edge_sh_normalization
        )
        # output edge diff irreps
        self.irreps_out = edge_sh_irreps

    def forward(self, data: AtomsData):
        edge_diff_embedding = self.sh(
            data.edge_vectors
        )
        data = replace_properties(data, edge_diff_embedding=edge_diff_embedding)
        
        return data

# Based on mir-group/nequip
def tp_out_irreps_with_instructions(
    irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps
) -> Tuple[o3.Irreps, List]:
    trainable = True

    # Collect possible irreps and their instructions
    irreps_out_list: List[Tuple[int, o3.Irreps]] = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in target_irreps:
                    k = len(irreps_out_list)  # instruction index
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", trainable))

    # We sort the output irreps of the tensor product so that we can simplify them
    # when they are provided to the second o3.Linear
    irreps_out = o3.Irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()

    # Permute the output indexes of the instructions to match the sorted irreps:
    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]

    instructions = sorted(instructions, key=lambda x: x[2])

    return irreps_out, instructions

#@compile_mode("script")
class reshape_irreps(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.dims = []
        self.muls = []
        for mul, ir in self.irreps:
            d = ir.dim
            self.dims.append(d)
            self.muls.append(mul)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        ix = 0
        out = []
        batch, _ = tensor.shape
        for mul, d in zip(self.muls, self.dims):
            field = tensor[:, ix : ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d
            field = field.reshape(batch, mul, d)
            out.append(field)
        return torch.cat(out, dim=-1)

#@torch.jit.script
def scatter_add(
    x: torch.Tensor, index: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    y = tmp.index_add(dim, index, x)
    return y

#@compile_mode("script")
class RealAgnosticResidualInteractionBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in, 
        target_irreps,
        hidden_irreps,
        avg_num_neighbors: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.irreps_in = irreps_in
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self._initialized = True if avg_num_neighbors is not None else False
        avg_num_neighbors = torch.ones((1,)) if avg_num_neighbors is None else torch.tensor([avg_num_neighbors])
        self.register_buffer("avg_num_neighbors", avg_num_neighbors) 

        # First linear
        self.linear_1 = o3.Linear(
            self.irreps_in['node_feat'],
            self.irreps_in['node_feat'],
            internal_weights=True,
            shared_weights=True,
        )
        
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.irreps_in['node_feat'],
            self.irreps_in['edge_diff_embedding'],
            self.target_irreps,
        )
        
        self.conv_tp = o3.TensorProduct(
            self.irreps_in['node_feat'],
            self.irreps_in['edge_diff_embedding'],
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.irreps_in['edge_dist_embedding'].num_irreps
        self.conv_tp_weights = FullyConnectedNet(
            [input_dim] + 3 * [64] + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        self.linear_2 = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.irreps_in['node_feat'], 
            self.irreps_in['node_attr'],
            self.hidden_irreps,
        )
        self.reshape = reshape_irreps(self.irreps_out)

    def forward(
        self,
        node_feat, 
        node_attr,
        edge_idx, 
        edge_dist_embedding,
        edge_diff_embedding,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:      
        sc = self.skip_tp(node_feat, node_attr)
        node_feat = self.linear_1(node_feat)
        tp_weights = self.conv_tp_weights(edge_dist_embedding)
        edge_feat = self.conv_tp(
            node_feat[edge_idx[:, 0]],
            edge_diff_embedding,
            tp_weights,
        )
        node_feat = scatter_add(
            edge_feat, edge_idx[:, 0], dim_size=len(node_feat), dim=0
        )
        node_feat = self.linear_2(node_feat)
        node_feat = node_feat / self.avg_num_neighbors
        
        return (self.reshape(node_feat), sc)
    
    def datamodule(self, _datamodule):
        if not self._initialized:
            avg_num_neigh = _datamodule._get_avg_num_neighbors()
            if avg_num_neigh is not None:
                self.avg_num_neighbors = torch.tensor([avg_num_neigh])

_TP = collections.namedtuple("_TP", "op, args")
_INPUT = collections.namedtuple("_INPUT", "tensor, start, stop")

def _wigner_nj(
    irrepss: List[o3.Irreps],
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irrepss = [o3.Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid is not None:
        filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]

    if len(irrepss) == 1:
        (irreps,) = irrepss
        ret = []
        e = torch.eye(irreps.dim, dtype=dtype)
        i = 0
        for mul, ir in irreps:
            for _ in range(mul):
                sl = slice(i, i + ir.dim)
                ret += [(ir, _INPUT(0, sl.start, sl.stop), e[sl])]
                i += ir.dim
        return ret

    *irrepss_left, irreps_right = irrepss
    ret = []
    for ir_left, path_left, C_left in _wigner_nj(
        irrepss_left,
        normalization=normalization,
        filter_ir_mid=filter_ir_mid,
        dtype=dtype,
    ):
        i = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                if filter_ir_mid is not None and ir_out not in filter_ir_mid:
                    continue

                C = o3.wigner_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype)
                if normalization == "component":
                    C *= ir_out.dim**0.5
                if normalization == "norm":
                    C *= ir_left.dim**0.5 * ir.dim**0.5

                C = torch.einsum("jk,ijl->ikl", C_left.flatten(1), C)
                C = C.reshape(
                    ir_out.dim, *(irreps.dim for irreps in irrepss_left), ir.dim
                )
                for u in range(mul):
                    E = torch.zeros(
                        ir_out.dim,
                        *(irreps.dim for irreps in irrepss_left),
                        irreps_right.dim,
                        dtype=dtype,
                    )
                    sl = slice(i + u * ir.dim, i + (u + 1) * ir.dim)
                    E[..., sl] = C
                    ret += [
                        (
                            ir_out,
                            _TP(
                                op=(ir_left, ir, ir_out),
                                args=(
                                    path_left,
                                    _INPUT(len(irrepss_left), sl.start, sl.stop),
                                ),
                            ),
                            E,
                        )
                    ]
            i += mul * ir.dim
    return sorted(ret, key=lambda x: x[0])


def U_matrix_real(
    irreps_in: Union[str, o3.Irreps],
    irreps_out: Union[str, o3.Irreps],
    correlation: int,
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irreps_out = o3.Irreps(irreps_out)
    irrepss = [o3.Irreps(irreps_in)] * correlation
    if correlation == 4:
        filter_ir_mid = [
            o3.Irrep(0, 1),
            o3.Irrep(1, -1),
            o3.Irrep(2, 1),
            o3.Irrep(3, -1),
            o3.Irrep(4, 1),
            o3.Irrep(5, -1),
            o3.Irrep(6, 1),
            o3.Irrep(7, -1),
            o3.Irrep(8, 1),
            o3.Irrep(9, -1),
            o3.Irrep(10, 1),
            o3.Irrep(11, -1),
        ]
    wigners = _wigner_nj(irrepss, normalization, filter_ir_mid, dtype)
    current_ir = wigners[0][0]
    out = []
    stack = torch.tensor([])
    for ir, _, base_o3 in wigners:
        if ir in irreps_out and ir == current_ir:
            stack = torch.cat((stack, base_o3.squeeze().unsqueeze(-1)), dim=-1)
            last_ir = current_ir
        elif ir in irreps_out and ir != current_ir:
            if len(stack) != 0:
                out += [last_ir, stack]
            stack = base_o3.squeeze().unsqueeze(-1)
            current_ir, last_ir = ir, ir
        else:
            current_ir = ir
    out += [last_ir, stack]
    return out

BATCH_EXAMPLE = 10
ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]

class Contraction(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irrep_out: o3.Irreps,
        correlation: int,
        internal_weights: bool = True,
        num_elements: Optional[int] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.num_features = irreps_in.count((0, 1))
        self.coupling_irreps = o3.Irreps([irrep.ir for irrep in irreps_in])
        self.correlation = correlation
        dtype = torch.get_default_dtype()
        for nu in range(1, correlation + 1):
            U_matrix = U_matrix_real(
                irreps_in=self.coupling_irreps,
                irreps_out=irrep_out,
                correlation=nu,
                dtype=dtype,
            )[-1]
            self.register_buffer(f"U_matrix_{nu}", U_matrix)

        # Tensor contraction equations
        self.contractions_weighting = torch.nn.ModuleList()
        self.contractions_features = torch.nn.ModuleList()

        # Create weight for product basis
        self.weights = torch.nn.ParameterList([])

        for i in range(correlation, 0, -1):
            # Shapes definying
            num_params = self.U_tensors(i).size()[-1]
            num_equivariance = 2 * irrep_out.lmax + 1
            num_ell = self.U_tensors(i).size()[-2]

            if i == correlation:
                parse_subscript_main = (
                    [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                    + ["ik,ekc,bci,be -> bc"]
                    + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                )
                graph_module_main = torch.fx.symbolic_trace(
                    lambda x, y, w, z: torch.einsum(
                        "".join(parse_subscript_main), x, y, w, z
                    )
                )

                # Optimizing the contractions
                self.graph_opt_main = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_main,
                    example_inputs=(
                        torch.randn(
                            [num_equivariance] + [num_ell] * i + [num_params]
                        ).squeeze(0),
                        torch.randn((num_elements, num_params, self.num_features)),
                        torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                        torch.randn((BATCH_EXAMPLE, num_elements)),
                    ),
                )
                # Parameters for the product basis
                w = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, self.num_features))
                    / num_params
                )
                self.weights_max = w
            else:
                # Generate optimized contractions equations
                parse_subscript_weighting = (
                    [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                    + ["k,ekc,be->bc"]
                    + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                )
                parse_subscript_features = (
                    ["bc"]
                    + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                    + ["i,bci->bc"]
                    + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                )

                # Symbolic tracing of contractions
                graph_module_weighting = torch.fx.symbolic_trace(
                    lambda x, y, z: torch.einsum(
                        "".join(parse_subscript_weighting), x, y, z
                    )
                )
                graph_module_features = torch.fx.symbolic_trace(
                    lambda x, y: torch.einsum("".join(parse_subscript_features), x, y)
                )

                # Optimizing the contractions
                graph_opt_weighting = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_weighting,
                    example_inputs=(
                        torch.randn(
                            [num_equivariance] + [num_ell] * i + [num_params]
                        ).squeeze(0),
                        torch.randn((num_elements, num_params, self.num_features)),
                        torch.randn((BATCH_EXAMPLE, num_elements)),
                    ),
                )
                graph_opt_features = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_features,
                    example_inputs=(
                        torch.randn(
                            [BATCH_EXAMPLE, self.num_features, num_equivariance]
                            + [num_ell] * i
                        ).squeeze(2),
                        torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                    ),
                )
                self.contractions_weighting.append(graph_opt_weighting)
                self.contractions_features.append(graph_opt_features)
                # Parameters for the product basis
                w = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, self.num_features))
                    / num_params
                )
                self.weights.append(w)
        if not internal_weights:
            self.weights = weights[:-1]
            self.weights_max = weights[-1]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        out = self.graph_opt_main( # the first layer for A features
            self.U_tensors(self.correlation), 
            self.weights_max,
            x,
            y,
        )
        for i, (weight, contract_weights, contract_features) in enumerate(
            zip(self.weights, self.contractions_weighting, self.contractions_features)
        ):
            c_tensor = contract_weights(  # other layers for A features
                self.U_tensors(self.correlation - i - 1), 
                weight,
                y,
            )
            c_tensor = c_tensor + out 
            out = contract_features(c_tensor, x) # B features
        resize_shape = torch.prod(torch.tensor(out.shape[1:]))
        return out.view(out.shape[0], resize_shape)

    def U_tensors(self, nu: int):
        return dict(self.named_buffers())[f"U_matrix_{nu}"]

class SymmetricContraction(CodeGenMixin, torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: Union[int, Dict[str, int]],
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        num_elements: Optional[int] = None,
    ) -> None:
        super().__init__()

        if irrep_normalization is None:
            irrep_normalization = "component"

        if path_normalization is None:
            path_normalization = "element"

        assert irrep_normalization in ["component", "norm", "none"]
        assert path_normalization in ["element", "path", "none"]

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)

        del irreps_in, irreps_out

        if not isinstance(correlation, tuple):
            corr = correlation
            correlation = {}
            for irrep_out in self.irreps_out:
                correlation[irrep_out] = corr

        assert shared_weights or not internal_weights

        if internal_weights is None:
            internal_weights = True

        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        del internal_weights, shared_weights

        self.contractions = torch.nn.ModuleList()
        for irrep_out in self.irreps_out:
            self.contractions.append(
                Contraction(
                    irreps_in=self.irreps_in,
                    irrep_out=o3.Irreps(str(irrep_out.ir)),
                    correlation=correlation[irrep_out],
                    internal_weights=self.internal_weights,
                    num_elements=num_elements,
                    weights=self.shared_weights,
                )
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        outs = [contraction(x, y) for contraction in self.contractions]
        return torch.cat(outs, dim=-1)


class EquivariantProductBasisBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        correlation: int,
        use_sc: bool = True,
        num_elements: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.use_sc = use_sc
        self.symmetric_contractions = SymmetricContraction(
            irreps_in=node_feats_irreps,
            irreps_out=target_irreps,
            correlation=correlation,
            num_elements=num_elements,
        )
        # Update linear
        self.linear = o3.Linear(
            target_irreps,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        sc: Optional[torch.Tensor],
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        node_feats = self.symmetric_contractions(node_feats, node_attrs)
        if self.use_sc and sc is not None:  # Update layer
            return self.linear(node_feats) + sc

        return self.linear(node_feats)

# @compile_mode('script')
class MACE(nn.Module):
    def __init__(
        self,
        cutoff: float,
        num_interactions: int,
        correlation: Union[int, List[int]],
        species: List[str],
        num_elements: Optional[int] = None,
        hidden_irreps: Union[o3.Irreps, str, None] = None,
        edge_sh_irreps: Union[o3.Irreps, str, None] = None,
        node_irreps: Union[o3.Irreps, str, None] = None,
        MLP_irreps: Union[o3.Irreps, str, None] = None,
        avg_num_neighbors: Optional[float] = None,
        lmax: int = 2,
        parity: bool = True,
        num_features: Optional[int] = None,
        num_basis: int = 8,
        power: int = 6,
        gate: Union[str, Callable] = 'silu',
        **kwargs,
    ) -> None:
        """
        Args:
            cutoff (float): Cutoff radius
            num_interactions (int): Number of interaction blocks
            correlation (int): Correlation type. 0 for dot product, 1 for cosine similarity
            species (List[str]): List of species
            num_elements (Optional[int], optional): Number of elements. Defaults to None.
            hidden_irreps (Union[o3.Irreps, str, None], optional): Hidden irreps. Defaults to None.
            edge_sh_irreps (Union[o3.Irreps, str, None], optional): Edge irreps. Defaults to None.
            node_irreps (Union[o3.Irreps, str, None], optional): Node irreps. Defaults to None.
            MLP_irreps (Union[o3.Irreps, str, None], optional): MLP irreps. Defaults to None.
            avg_num_neighbors (Optional[float], optional): Average number of neighbors. Defaults to None.
            lmax (int, optional): Maximum l value. Defaults to 2.
            parity (bool, optional): Parity. Defaults to True.
            num_features (Optional[int], optional): Number of features. Defaults to None.
            num_basis (int, optional): Number of radial basis. Defaults to 8.
            power (int, optional): Power of radial basis. Defaults to 6.
            gate (Union[str, Callable], optional): Activation function for gate. Defaults to 'silu'.
        """
        super().__init__()
        
        self.cutoff = cutoff
        self.lmax = lmax
        self.parity = parity
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions

        if num_elements is None:
            num_elements = len(species) if species is not None else 119
        
        # hidden feature irreps
        if hidden_irreps is not None:
            self.hidden_irreps = o3.Irreps(hidden_irreps) if isinstance(hidden_irreps, str) else hidden_irreps
        else:
            self.hidden_irreps = o3.Irreps(
                [
                    (num_features, o3.Irrep(l, p))
                    for p in ((1, -1) if parity else (1,))
                    for l in range(lmax + 1)
                ]
            )
        # MACE prohibits some irreps like 0e, 1e to be used
        forbidden_ir = ['0o', '1e', '2o', '3e', '4o']
        self.hidden_irreps = o3.Irreps([irrep for irrep in self.hidden_irreps if str(irrep.ir) not in forbidden_ir])
        self.num_features = self.hidden_irreps.count(o3.Irrep(0, 1))

        ## handling irreps
        # chemical embedding irreps
        if node_irreps is None:
            self.node_irreps = o3.Irreps([(self.num_features, o3.Irrep(0, 1))])
        elif isinstance(node_irreps, str):
            self.node_irreps = o3.Irreps(node_irreps)
        else:
            self.node_irreps = node_irreps
        # edge sphere harmonic irreps
        if edge_sh_irreps is None:
            self.edge_sh_irreps = o3.Irreps.spherical_harmonics(lmax, p=-1 if parity else 1)
        elif isinstance(edge_sh_irreps, str):
            self.edge_sh_irreps = o3.Irreps(edge_sh_irreps)
        else:
            self.edge_sh_irreps = edge_sh_irreps
        
        # MLP_irreps
        if MLP_irreps is None:
            self.MLP_irreps = o3.Irreps([(max(1, self.num_features // 2), o3.Irrep(0, 1))])
        elif isinstance(MLP_irreps, str):
            self.MLP_irreps = o3.Irreps(MLP_irreps)
        else:
            self.MLP_irreps = MLP_irreps
            
        self.embeddings = nn.ModuleDict()
        self.embeddings['onehot_embedding'] = OneHotAtomEncoding(num_elements=num_elements, species=species)
        self.embeddings['radial_basis'] = RadialBasisEdgeEncoding(
            basis=BesselBasis(cutoff=cutoff, num_basis=num_basis),
            cutoff_fn=PolynomialCutoff(cutoff=cutoff, power=power),
        )
        self.embeddings['sphere_harmonics'] = SphericalHarmonicEdgeAttrs(edge_sh_irreps=self.edge_sh_irreps)
        
        self.irreps_in = {
            'edge_diff_embedding': self.embeddings.sphere_harmonics.irreps_out,
            'edge_dist_embedding': self.embeddings.radial_basis.irreps_out,
        }
        self.irreps_in.update(self.embeddings.onehot_embedding.irreps_out)

        self.embeddings['chemical_embedding'] = AtomwiseLinear(
            irreps_in=self.irreps_in['node_attr'],
            irreps_out=self.node_irreps,
        )
        self.irreps_in['node_feat'] = self.embeddings.chemical_embedding.irreps_out
        
        interaction_irreps = (self.edge_sh_irreps * self.num_features).sort()[0].simplify()
        
        self.interactions = torch.nn.ModuleList()
        self.products = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()
        gate_fn = activation_fn[gate] if isinstance(gate, str) else gate
        # interaction blocks
        for i in range(num_interactions):
            hidden_irreps_out = str(self.hidden_irreps[0]) if i == num_interactions - 1 else self.hidden_irreps
            if i > 0:
                self.irreps_in['node_feat'] = self.hidden_irreps
            inter = RealAgnosticResidualInteractionBlock(
                irreps_in=self.irreps_in,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
            )
            self.interactions.append(inter)
            
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=inter.target_irreps if i == 0 else interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i],
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            
            if i == num_interactions - 1:
                readout = AtomwiseNonLinear(
                    irreps_in=hidden_irreps_out, 
                    MLP_irreps=self.MLP_irreps,
                    gate=gate_fn,
                )
            else:
                readout = o3.Linear(irreps_in=hidden_irreps_out, irreps_out=o3.Irreps('1x0e'))
            self.readouts.append(readout)

            self.atomwise_reduce = AtomwiseReduce(output_key='energy')

        self.compute_forces = False
        if 'compute_forces' in kwargs.keys():
            if kwargs['compute_forces']:
                self.compute_forces = True
                self.gradient_output = GradientOutput(model_outputs=['forces'])
            
    def forward(self, data: AtomsData):
        """
        Args:
            data (AtomsData): A NamedTuple of model inputs

        Returns:
            dict: A dictionary with keys 'energy' and 'forces'
        """

        for m in self.embeddings.values():
            data = m(data)
        
        node_es_list = []
        node_feat = data.node_feat
        node_attr = data.node_attr
        edge_indices = data.edge_indices
        edge_dist_embedding = data.edge_dist_embedding
        edge_diff_embedding = data.edge_diff_embedding
        assert node_feat is not None
        assert node_attr is not None
        assert edge_dist_embedding is not None
        assert edge_diff_embedding is not None
        
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feat, sc = interaction(
                node_feat, 
                node_attr,
                edge_indices, 
                edge_dist_embedding,
                edge_diff_embedding,
            )
            node_feat = product(
                node_feats=node_feat,
                sc=sc,
                node_attrs=node_attr,
            )
            node_es_list.append(readout(node_feat).squeeze())
        
        node_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        data = replace_properties(data, atomic_energy=node_es)

        data = self.atomwise_reduce(data)
        
        if self.compute_forces:
            data = self.gradient_output(data)

        return data
    
class AtomwiseReduce(nn.Module):
    def __init__(
        self,
        output_key: str = "energy",
        per_atom_output: bool = False,
        aggregation_mode: str = "sum",     # should be sum or mean
    ) -> None:
        super().__init__()
        # self.model_outputs = [output_key]
        # if per_atom_output:
        #     self.model_outputs.append(output_key + '_per_atom')
        self.aggregation_mode = aggregation_mode
        self.per_atom_output = per_atom_output
    
    def forward(self, data: AtomsData):
        y = torch.zeros_like(
            data.num_atoms, 
            dtype=data.edge_vectors.dtype,
            device=data.edge_vectors.device
        )
        atomic_energy = data.atomic_energy
        assert atomic_energy is not None
        image_indices = data.image_indices
        if image_indices is None:
            image_indices = torch.zeros_like(atomic_energy, dtype=torch.long)
        assert image_indices is not None
        y.index_add_(0, image_indices, atomic_energy)
        
        if self.aggregation_mode == "mean":
            y = y / data.num_atoms
        
        data = replace_properties(data, energy=y)
        if self.per_atom_output:
            assert atomic_energy is not None
            data = replace_properties(data, atomic_energy=atomic_energy)
        return data

class GradientOutput(torch.nn.Module):
    def __init__(
        self,
        grad_on_edge_diff: bool = True,
        grad_on_positions: bool = False,
        model_outputs: List[str] = ['forces'],       # properties that need to be calculated, can be forces, stress, virial, etc.
        update_callback: Optional[Callable] = None,  # Add a callback parameter
    ) -> None:
        super().__init__()
        self.grad_on_edge_diff = grad_on_edge_diff
        self.grad_on_positions = grad_on_positions
        self.update_callback = update_callback
        self.model_outputs = model_outputs

    #@torch.jit.ignore
    def update_model_outputs(self, outputs: Union[List[str], str]):
        if isinstance(outputs, str):
            self.model_outputs.append(outputs)
        else:
            self.model_outputs.extend(outputs)
        if self.update_callback:
            self.update_callback()

    def forward(self, data: AtomsData, training: bool=True,)->AtomsData:
        if self.grad_on_edge_diff:
            energy = data.energy
            edge_vectors = data.edge_vectors
            positions = data.positions
            edge_indices = data.edge_indices
            assert energy is not None
            
            if 'forces' in self.model_outputs:
                # edge_vectors.requires_grad_()
                outputs_list = torch.jit.annotate(List[torch.Tensor], [energy])
                inputs_list = torch.jit.annotate(List[torch.Tensor], [edge_vectors])
                grad_outputs_list = torch.jit.annotate(Optional[List[Optional[torch.Tensor]]], [torch.ones_like(energy)])
                dE_ddiff = torch.autograd.grad(
                    outputs=outputs_list,
                    inputs=inputs_list,
                    grad_outputs=grad_outputs_list,
                    retain_graph=training,
                    create_graph=training,
                )[0]

                # Initialize forces with proper strides
                assert dE_ddiff is not None
                i_forces = torch.zeros(positions.shape[0], 3, device=positions.device, dtype=positions.dtype)
                j_forces = torch.zeros(positions.shape[0], 3, device=positions.device, dtype=positions.dtype)
                i_forces.index_add_(0, edge_indices[:, 0], dE_ddiff)
                j_forces.index_add_(0, edge_indices[:, 1], -dE_ddiff)
                forces = i_forces + j_forces
                
                data = replace_properties(data, forces=forces)

                # Reference: https://en.wikipedia.org/wiki/Virial_stress
                # This method calculates virials by giving pair-wise force components
                
        #         if 'stress' in self.model_outputs or 'virial' in self.model_outputs:
        #             image_indices = data.image_indices
        #             atomic_virial = torch.einsum("ij, ik -> ijk", edge_vectors, dE_ddiff)           # I'm quite not sure if a negative sign should be added before dE_ddiff, but I think it should be right
        #             # stress = torch.zeros_like(cell).index_add(0, , atomic_stress)
        #             atomic_virial = torch.zeros(
        #                 (forces_dim, 3, 3),                                         
        #                 dtype=forces.dtype,
        #                 # it seens like the calculation is not very right... because f_ij is not absolutely right here. Maybe we need to do something like in force calculation
        #                 # add i_stress and j_stress together then it is the total stress. need verification
        #                 device=forces.device).index_add(0, edge_idx[:, 0], atomic_virial)
        #             # j_stress = torch.zeros_like(i_stress).index_add(0, edge_idx[:, 1], -atomic_stress)
        #             # atomic_stress = i_stress + j_stress          
        #             virial = torch.zeros(
        #                 energy.shape[0], 3, 3, 
        #                 dtype=forces.dtype, 
        #                 device=forces.device).index_add(0, image_indices, atomic_virial)  # don't need to divide by two
        #             data["virial"] = virial.view(-1, 9)[:, [0, 4, 8, 5, 2, 1]]
        #             if "cell" in data and 'stress' in self.model_outputs:
        #                 cell = data.cell.view(-1, 3, 3)
        #                 volumes = torch.sum(cell[:, 0] * cell[:, 1].cross(cell[:, 2], dim=-1), dim=1)
        #                 stress = - virial / volumes[:, None, None]
        #                 data["stress"] = stress.view(-1, 9)[:, [0, 4, 8, 5, 2, 1]]
            
        # elif self.grad_on_positions:
        #     energy = data.energy
        #     grad_outputs : List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
        #     if 'forces' in self.model_outputs:
        #         grad_inputs = [data.positions]
        #         if 'stress' in self.model_outputs:
        #             grad_inputs.append(data["strain"])
        #         grads = torch.autograd.grad(
        #             [energy,],
        #             grad_inputs,
        #             grad_outputs=grad_outputs,
        #             retain_graph=training,
        #             create_graph=training,
        #         )
        #         dEdR = grads[0]
        #         if dEdR is None:
        #             dEdR = torch.zeros_like(data.positions)
        #         data.forces = -dEdR
                    
        #         if 'stress' in self.model_outputs:
        #             if "cell" in data:
        #                 stress = grads[1]
        #                 if stress is None:
        #                     stress = torch.zeros_like(data.cell)
        #                 cell = data.cell.view(-1, 3, 3)
        #                 volumes = torch.sum(cell[:, 0] * cell[:, 1].cross(cell[:, 2], dim=-1), dim=1)
        #                 stress /= volumes[:, None, None]
        #                 data["stress"] = stress.view(-1, 9)[:, [0, 4, 8, 5, 2, 1]] 
        
        # else:
        #     raise ValueError("Gradients must be calculated with respect to positions or R_ij. Nothing is given!")
                    
        return data