from iann.data import AtomsData, replace_properties
import torch
from torch import nn
import abc, math
from ase.data import atomic_numbers
from typing import Dict, List, Optional, Union, Callable
import warnings
import logging
logging.getLogger("cuequivariance").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="The TorchScript type system doesn't support instance-level annotations")
warnings.filterwarnings("ignore", message="cuequivariance_ops_torch is not available")
warnings.filterwarnings("ignore", message="Fused TP is not supported on CPU")

from e3nn import o3
from e3nn.o3 import Linear, TensorProduct, FullyConnectedTensorProduct
from e3nn.nn import FullyConnectedNet, Gate, NormActivation

import torch.compiler
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False

# Try to import cuEquivariance, fallback to e3nn if not available
try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet
    CUEQUIVARIANCE_AVAILABLE = True
    warnings.warn("cuEquivariance detected - using optimized operations", UserWarning)
except (ImportError, SyntaxError, Exception) as e:
    warnings.warn(f"cuEquivariance not available - falling back to e3nn (Warning: {e})", UserWarning)
    CUEQUIVARIANCE_AVAILABLE = False

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
        
    def forward(self, data: AtomsData) -> torch.Tensor:
        atomic_types = self.transform(data.atomic_numbers)
        assert torch.all(atomic_types >= 0), "Provided data contains species not defined in TypeMapper!"
        return atomic_types
        
    def transform(self, numbers: torch.Tensor) -> torch.Tensor:
        if numbers.max() > 119 or numbers.min() < 1:
            raise ValueError("Provided atomic numbers are not in the periodic table!")
        types = self.Z_to_index[numbers]
        return types
    
    def untransform(self, types: torch.Tensor) -> torch.Tensor:
        return self.index_to_Z[types]


class OneHotAtomEncoding(torch.nn.Module):
    """Copmute a one-hot floating point encoding of atoms' discrete atom types.

    Args:
        set_features: If ``True`` (default), ``node_features`` will be set in addition to ``node_attrs``.
    """

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
            'node_attr': o3.Irreps([(self.num_elements, o3.Irrep(0, 1))])
        }
        if self.set_features:
            self.irreps_out['node_feat'] = self.irreps_out['node_attr']
            
    def forward(self, data: AtomsData) -> AtomsData:
        if self.type_mapper is not None:
            atomic_types = self.type_mapper(data)
        else:
            atomic_types = data.atomic_numbers - 1
    
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
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
    ):
        super().__init__()
        self.irreps_in: o3.Irreps = irreps_in
        self.irreps_out: o3.Irreps = irreps_out
        
        if CUEQUIVARIANCE_AVAILABLE:
            self.linear = cuet.Linear(
                irreps_in=cue.Irreps(cue.O3, self.irreps_in),
                irreps_out=cue.Irreps(cue.O3, self.irreps_out),
                layout=cue.mul_ir
            )
        else:
            self.linear = o3.Linear(
                irreps_in=self.irreps_in, irreps_out=self.irreps_out
            )

    def forward(self, data: AtomsData) -> AtomsData:
        if data.node_feat is None:
            raise RuntimeError("node_feat must not be None")
        node_feat = data.node_feat
        assert isinstance(node_feat, torch.Tensor)
        node_feat = self.linear(node_feat)
        data = replace_properties(data, node_feat=node_feat)
        return data

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

class CutoffFunction(torch.nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self):
        pass

def _poly_cutoff(x: torch.Tensor, factor: float, p: float = 6.0) -> torch.Tensor:
    x = x * factor

    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
    out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))

    return out * (x < 1.0)

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
    
class RadialBasisEdgeEncoding(torch.nn.Module):
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

    def forward(self, data: AtomsData) -> AtomsData:
        edge_diff = data.edge_vectors
        edge_dist = torch.linalg.norm(edge_diff, dim=1)
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

    def forward(self, data: AtomsData) -> AtomsData:
        edge_diff_embedding = self.sh(
            data.edge_vectors
        )
        data = replace_properties(data, edge_diff_embedding=edge_diff_embedding)
        return data

def scatter_add(
    x: torch.Tensor, index: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    y = tmp.index_add(dim, index, x)
    return y

def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False

class ConvNetLayer(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        irreps_out,
        invariant_layers: int=1,
        invariant_neurons: int=8,
        avg_num_neighbors: Optional[float]=None,
        use_sc: bool=True,
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp"},
    ) -> None:
        """
        Convolution Block.

        :param irreps_in: Input irreps, including 
        :param irreps_out: Output irreps, in our case typically a single scalar
        :param radial_layers: Number of radial layers, default = 1
        :param radial_neurons: Number of hidden neurons in radial function, default = 8
        :param avg_num_neighbors: Number of neighbors to divide by, default None => no normalization.
        :param number_of_basis: Number or Basis function, default = 8
        :param irreps_in: Input Features, default = None
        :param use_sc: bool, use self-connection or not
        """
        super().__init__()

        if avg_num_neighbors is not None:
            self._initialized = True
            avg_num_neigh = torch.tensor([avg_num_neighbors])
        else:
            self._initialized = False
            avg_num_neigh = torch.ones((1,))
        
        # self._initialized = True if avg_num_neighbors is not None else False
        # avg_num_neighbors = torch.ones((1,)) if avg_num_neighbors is None else torch.tensor([avg_num_neighbors])
        self.register_buffer("avg_num_neighbors", avg_num_neigh)
        self.use_sc = use_sc

        feature_irreps_in = irreps_in['node_feat']
        feature_irreps_out = irreps_out
        edge_diff_irreps = irreps_in['edge_diff_embedding']
        edge_dist_irreps = irreps_in['edge_dist_embedding']

        # - Build modules -
        if CUEQUIVARIANCE_AVAILABLE:
            self.linear_1 = cuet.Linear(
                irreps_in=cue.Irreps(cue.O3, feature_irreps_in),
                irreps_out=cue.Irreps(cue.O3, feature_irreps_in),
                layout=cue.mul_ir,
                internal_weights=True,
                shared_weights=True,
            )
        else:
            self.linear_1 = o3.Linear(
                irreps_in=feature_irreps_in,
                irreps_out=feature_irreps_in,
                internal_weights=True,
                shared_weights=True,
            )

        irreps_mid = []
        instructions = []

        for i, (mul, ir_in) in enumerate(feature_irreps_in):
            for j, (_, ir_edge) in enumerate(edge_diff_irreps):
                for ir_out in ir_in * ir_edge:
                    if ir_out in feature_irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, o3.Irrep(ir_out)))
                        instructions.append((i, j, k, "uvu", True))

        # We sort the output irreps of the tensor product so that we can simplify them
        # when they are provided to the second o3.Linear
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        # Permute the output indexes of the instructions to match the sorted irreps:
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        if CUEQUIVARIANCE_AVAILABLE:
            tp = cuet.ChannelWiseTensorProduct(
                irreps_in1=cue.Irreps(cue.O3, feature_irreps_in),
                irreps_in2=cue.Irreps(cue.O3, edge_diff_irreps),
                layout=cue.mul_ir,
                filter_irreps_out=None,
                shared_weights=False,
                internal_weights=False,
            )
        else:
            tp = o3.TensorProduct(
                feature_irreps_in,
                edge_diff_irreps,
                irreps_mid,
                instructions,
                shared_weights=False,
                internal_weights=False,
            )

        # init_irreps already confirmed that the edge embeddding is all invariant scalars
        self.fc = FullyConnectedNet(
            [edge_dist_irreps.num_irreps]
            + invariant_layers * [invariant_neurons]
            + [tp.weight_numel],
            {
                "ssp": ShiftedSoftPlus,
                "silu": torch.nn.functional.silu,
            }[nonlinearity_scalars["e"]],
        )

        self.tp = tp

        if CUEQUIVARIANCE_AVAILABLE:
            self.linear_2 = cuet.Linear(
                irreps_in=cue.Irreps(cue.O3, irreps_mid.simplify()),
                irreps_out=cue.Irreps(cue.O3, feature_irreps_out),
                layout=cue.mul_ir,
                internal_weights=True,
                shared_weights=True,
            )
        else:
            self.linear_2 = o3.Linear(
                irreps_in=irreps_mid.simplify(),
                irreps_out=feature_irreps_out,
                internal_weights=True,
                shared_weights=True,
            )

        self.sc = None
        if self.use_sc:
            if CUEQUIVARIANCE_AVAILABLE:
                self.sc = cuet.FullyConnectedTensorProduct(
                    irreps_in1=cue.Irreps(cue.O3, feature_irreps_in),
                    irreps_in2=cue.Irreps(cue.O3, irreps_in['node_attr']),
                    irreps_out=cue.Irreps(cue.O3, feature_irreps_out),
                    layout=cue.mul_ir,
                )
            else:
                self.sc = o3.FullyConnectedTensorProduct(
                    feature_irreps_in,
                    irreps_in['node_attr'],
                    feature_irreps_out,
                )

    def forward(self, data: AtomsData) -> AtomsData:
        """
        Evaluate interaction Block with ResNet (self-connection).

        :param node_input:
        :param node_attr:
        :param edge_src:
        :param edge_dst:
        :param edge_attr:
        :param edge_length_embedded:

        :return:
        """
        node_feat = data.node_feat
        assert node_feat is not None
        node_attr = data.node_attr
        assert node_attr is not None
        edge_dist_embedding = data.edge_dist_embedding
        assert edge_dist_embedding is not None
        edge_diff_embedding = data.edge_diff_embedding
        assert edge_diff_embedding is not None
        edge_indices = data.edge_indices

        weight = self.fc(edge_dist_embedding)

        x = node_feat
        if self.sc is not None:
            sc = self.sc(x, node_attr)

        x = self.linear_1(x)
        edge_features = self.tp(
            x[edge_indices[:, 1]], edge_diff_embedding, weight
        )
        x = scatter_add(edge_features, edge_indices[:, 0], dim_size=len(x), dim=0)

        # Necessary to get TorchScript to be able to type infer when its not None
        # avg_num_neigh: Optional[float] = self.avg_num_neighbors
        # if avg_num_neigh is not None:
        x = x.div(self.avg_num_neighbors**0.5)

        x = self.linear_2(x)

        if self.sc is not None:
            x = x + sc

        node_feat = x
        data = replace_properties(data, node_feat=node_feat)
        return data
    
    def datamodule(self, _datamodule):
        if not self._initialized:
            avg_num_neigh = _datamodule._get_avg_num_neighbors()
            if avg_num_neigh is not None:
                self.avg_num_neighbors = torch.tensor([avg_num_neigh])


def ShiftedSoftPlus(x):
    return torch.nn.functional.softplus(x) - torch.log(torch.tensor(2.0))

def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False

acts = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "ssp": ShiftedSoftPlus,
    "silu": torch.nn.functional.silu,
}

class InteractionLayer(torch.nn.Module):
    """
    Args:

    """

    def __init__(
        self,
        irreps_in,
        feature_irreps_hidden,
        convolution=ConvNetLayer,
        convolution_kwargs: dict = {},
        resnet: bool = False,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "ssp", "o": "abs"},
    ):
        super().__init__()
        # initialization
        assert nonlinearity_type in ("gate", "norm")
        # make the nonlin dicts from parity ints instead of convinience strs
        nonlinearity_scalars_dict = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        nonlinearity_gates_dict = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }

        self.feature_irreps_hidden = o3.Irreps(feature_irreps_hidden)
        self.resnet = resnet
        self.irreps_out = irreps_in.copy()

        self.irreps_in = irreps_in
        edge_diff_irreps = self.irreps_in['edge_diff_embedding']
        irreps_layer_out_prev = self.irreps_in['node_feat']

        irreps_scalars = o3.Irreps(
            [
                (mul, o3.Irrep(ir))
                for mul, ir in self.feature_irreps_hidden
                if ir.l == 0
                and tp_path_exists(irreps_layer_out_prev, edge_diff_irreps, ir)
            ]
        )

        irreps_gated = o3.Irreps(
            [
                (mul, o3.Irrep(ir))
                for mul, ir in self.feature_irreps_hidden
                if ir.l > 0
                and tp_path_exists(irreps_layer_out_prev, edge_diff_irreps, ir)
            ]
        )

        irreps_layer_out = (irreps_scalars + irreps_gated).simplify()

        if nonlinearity_type == "gate":
            ir = (
                "0e"
                if tp_path_exists(irreps_layer_out_prev, edge_diff_irreps, "0e")
                else "0o"
            )
            irreps_gates = o3.Irreps([(mul, o3.Irrep(ir)) for mul, _ in irreps_gated])

            # TO DO, it's not that safe to directly use the
            # dictionary
            equivariant_nonlin = Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=[
                    acts[nonlinearity_scalars_dict[ir.p]] for _, ir in irreps_scalars
                ],
                irreps_gates=irreps_gates,
                act_gates=[acts[nonlinearity_gates_dict[ir.p]] for _, ir in irreps_gates],
                irreps_gated=irreps_gated,
            )

            conv_irreps_out = equivariant_nonlin.irreps_in.simplify()

        else:
            conv_irreps_out = irreps_layer_out.simplify()

            equivariant_nonlin = NormActivation(
                irreps_in=conv_irreps_out,
                # norm is an even scalar, so use nonlinearity_scalars[1]
                scalar_nonlinearity=acts[nonlinearity_scalars_dict[1]],
                normalize=True,
                epsilon=1e-8,
                bias=False,
            )

        self.equivariant_nonlin = equivariant_nonlin

        if irreps_layer_out == irreps_layer_out_prev and resnet:
            self.resnet = True
        else:
            self.resnet = False

        # override defaults for irreps:
        convolution_kwargs.pop("irreps_in", None)
        convolution_kwargs.pop("irreps_out", None)
        self.conv = convolution(
            irreps_in=self.irreps_in,
            irreps_out=conv_irreps_out,
            nonlinearity_scalars=nonlinearity_scalars,
            **convolution_kwargs,
        )
        # output node feature irreps
        self.irreps_out['node_feat'] = self.equivariant_nonlin.irreps_out

    def forward(self, data: AtomsData):
        # save old features for resnet
        node_feat = data.node_feat
        assert node_feat is not None
        old_node_feat = node_feat.clone()
        # run convolution
        data = self.conv(data)
        # do nonlinearity
        node_feat = data.node_feat
        assert node_feat is not None
        node_feat = self.equivariant_nonlin(node_feat)
        data = replace_properties(data, node_feat=node_feat)
        if self.resnet:
            node_feat = node_feat + old_node_feat
            data = replace_properties(data, node_feat=node_feat)
        return data

class NequIP(torch.nn.Module):
    """
    A class to set up the NequIP model.
    """
    def __init__(
        self,
        num_layers: int,
        num_channels: int = 64,
        norm_data: bool = False,
        norm_per_atom: bool = False,
        data_stddev: float = 1.0,
        data_mean: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Initialize the NequIP model.
        """
        super().__init__()

        self.cutoff:float = kwargs.get('cutoff', 5.5)
        self.hidden_irreps:Union[o3.Irreps, str, None] = kwargs.get('hidden_irreps', None)
        self.edge_sh_irreps:Union[o3.Irreps, str, None] = kwargs.get('edge_sh_irreps', None)
        self.node_irreps:Union[o3.Irreps, str, None] = kwargs.get('node_irreps', None)
        self.MLP_irreps:Union[o3.Irreps, str, None] = kwargs.get('MLP_irreps', None)
        self.lmax:int = kwargs.get('lmax', 2)
        self.parity:bool = kwargs.get('parity', True)
        self.num_basis:int = kwargs.get('num_basis', 8)
        self.power:int = kwargs.get('power', 6)
        self.resnet:bool = kwargs.get('resnet', False)
        self.nonlinearity_type:str = kwargs.get('nonlinearity_type', "gate")
        self.nonlinearity_scalars:Dict[int, Callable] = kwargs.get('nonlinearity_scalars', {"e": "ssp", "o": "tanh"})
        self.nonlinearity_gates:Dict[int, Callable] = kwargs.get('nonlinearity_gates', {"e": "ssp", "o": "abs"})
        self.convolution_kwargs:dict = kwargs.get('convolution_kwargs', {})

        
        species: List[str] = kwargs.get('species', None)
        if bool(species):
            num_elements = len(species)
        else:
            num_elements = 119
        
        ## handling irreps
        # chemical embedding irreps
        if self.node_irreps is None:
            self.node_irreps = o3.Irreps([(num_channels, o3.Irrep(0, 1))])
        elif isinstance(self.node_irreps, str):
            self.node_irreps = o3.Irreps(self.node_irreps)
        else:
            self.node_irreps = self.node_irreps
        # edge sphere harmonic irreps
        if self.edge_sh_irreps is None:
            self.edge_sh_irreps = o3.Irreps.spherical_harmonics(self.lmax, p=-1 if self.parity else 1)
        elif isinstance(self.edge_sh_irreps, str):
            self.edge_sh_irreps = o3.Irreps(self.edge_sh_irreps)
        else:
            self.edge_sh_irreps = self.edge_sh_irreps
        # hidden feature irreps
        if self.hidden_irreps is None:
            self.hidden_irreps = o3.Irreps(
                [
                    (num_channels, o3.Irrep(l, p))
                    for p in ((1, -1) if self.parity else (1,))
                    for l in range(self.lmax + 1)
                ]
            )
        elif isinstance(self.hidden_irreps, str):
            self.hidden_irreps = o3.Irreps(self.hidden_irreps)
        else:
            self.hidden_irreps = self.hidden_irreps
        # MLP_irreps
        if self.MLP_irreps is None:
            self.MLP_irreps = o3.Irreps([(max(1, num_channels // 2), o3.Irrep(0, 1))])
        elif isinstance(self.MLP_irreps, str):
            self.MLP_irreps = o3.Irreps(self.MLP_irreps)
        else:
            self.MLP_irreps = self.MLP_irreps
        
        self.embeddings = nn.ModuleDict()
        self.embeddings['onehot_embedding'] = OneHotAtomEncoding(num_elements=num_elements, species=species)
        self.embeddings['radial_basis'] = RadialBasisEdgeEncoding(
            basis=BesselBasis(cutoff=self.cutoff, num_basis=self.num_basis),
            cutoff_fn=PolynomialCutoff(cutoff=self.cutoff, power=self.power),
        )
        self.embeddings['sphere_harmonics'] = SphericalHarmonicEdgeAttrs(edge_sh_irreps=self.edge_sh_irreps)
        
        self.irreps_in = {
            'edge_diff_embedding': self.embeddings.sphere_harmonics.irreps_out,
            'edge_dist_embedding': self.embeddings.radial_basis.irreps_out,
        }
        self.irreps_in.update(self.embeddings.onehot_embedding.irreps_out)
        
        self.embeddings['chemical_embedding'] = AtomwiseLinear(
            irreps_in=self.irreps_in['node_attr'], # from OneHotAtomEncoding
            irreps_out=self.node_irreps,
        )
        self.irreps_in['node_feat'] = self.embeddings.chemical_embedding.irreps_out
        
        self.interactions = nn.ModuleList()
        for _ in range(num_layers):
            interaction = InteractionLayer(
                irreps_in=self.irreps_in, 
                feature_irreps_hidden=self.hidden_irreps,
                convolution_kwargs=self.convolution_kwargs,
                resnet=self.resnet,
                nonlinearity_type=self.nonlinearity_type,
                nonlinearity_scalars=self.nonlinearity_scalars,
                nonlinearity_gates=self.nonlinearity_gates,
            )
            self.interactions.append(interaction)
            self.irreps_in.update(interaction.irreps_out)
        
        if CUEQUIVARIANCE_AVAILABLE:
            self.readout_mlp = nn.Sequential(
                cuet.Linear(
                    irreps_in=cue.Irreps(cue.O3, self.irreps_in['node_feat']),
                    irreps_out=cue.Irreps(cue.O3, self.MLP_irreps),
                    layout=cue.mul_ir
                ),
                cuet.Linear(
                    irreps_in=cue.Irreps(cue.O3, self.MLP_irreps), 
                    irreps_out=cue.Irreps(cue.O3, '1x0e'),
                    layout=cue.mul_ir
                ),
            )
        else:
            self.readout_mlp = nn.Sequential(
                o3.Linear(
                    irreps_in=self.irreps_in['node_feat'],
                    irreps_out=self.MLP_irreps,
                ),
                o3.Linear(
                    irreps_in=self.MLP_irreps, 
                    irreps_out=o3.Irreps('1x0e'),
                ),
            )

        # Normalisation constants
        self.norm_data = torch.nn.Parameter(torch.tensor(norm_data), requires_grad=False)
        self.norm_per_atom = torch.nn.Parameter(torch.tensor(norm_per_atom), requires_grad=False)
        self.data_stddev = torch.nn.Parameter(torch.tensor(data_stddev), requires_grad=False)
        self.data_mean = torch.nn.Parameter(torch.tensor(data_mean), requires_grad=False)
        
        self.atomwise_reduce = AtomwiseReduce(output_key='energy')
        
        self.compute_forces = False
        if 'compute_forces' in kwargs.keys():
            if kwargs['compute_forces']:
                self.compute_forces = True
                self.gradient_output = GradientOutput(model_outputs=['forces'])

    def forward(self, data: AtomsData):
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
        for m in self.embeddings.values():
            data = m(data)
            
        for m in self.interactions:
            data = m(data)
        
        node_feat = data.node_feat
        assert node_feat is not None
        atomic_energy = self.readout_mlp(node_feat).squeeze()
        assert atomic_energy is not None
        data = replace_properties(data, atomic_energy=atomic_energy)
        data = self.atomwise_reduce(data)

        # de-normalization
        if self.norm_data:
            normalizer = self.data_stddev
            energy = normalizer * data.energy
            mean_shift = self.data_mean
            if self.norm_per_atom:
                mean_shift = len(data.edge_indices) * mean_shift
            energy = energy + mean_shift
            data = replace_properties(data, energy=energy)

        if self.compute_forces:
            data = self.gradient_output(data)

        return data
    
    def get_optimization_info(self):
        """Get information about optimization status"""
        return {
            "cuequivariance_available": CUEQUIVARIANCE_AVAILABLE,
            "optimization_enabled": CUEQUIVARIANCE_AVAILABLE,
            "performance_boost": "2-5x speedup" if CUEQUIVARIANCE_AVAILABLE else "No optimization"
        }
     
class AtomwiseReduce(nn.Module):
    def __init__(
        self,
        output_key: str = "energy",
        per_atom_output: bool = False,
        aggregation_mode: str = "sum",     # should be sum or mean
    ) -> None:
        super().__init__()
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
        model_outputs: List[str] = ['forces'],    
        update_callback: Optional[Callable] = None,  # Add a callback parameter
    ) -> None:
        # TODO: define a set for allowed model outputs
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
        # update parent model
        if self.update_callback:
            self.update_callback()

    def forward(self, data: AtomsData, training: bool=True,):
        if self.grad_on_edge_diff:
            energy = data.energy
            edge_vectors = data.edge_vectors
            forces_dim = int(torch.sum(data.num_atoms))
            edge_indices = data.edge_indices
            assert energy is not None
            if 'forces' in self.model_outputs:
                grad_outputs : List[Optional[torch.Tensor]] = [torch.ones_like(energy)]    # for model deploy
                dE_ddiff = torch.autograd.grad(
                    [energy,],
                    [edge_vectors,],
                    grad_outputs=grad_outputs,
                    retain_graph=training,
                    create_graph=training,
                )
                dE_ddiff = torch.zeros_like(data.positions) if dE_ddiff is None else dE_ddiff[0]   # for torch.jit.script
                assert dE_ddiff is not None
                
                # diff = R_j - R_i, so -dE/dR_j = -dE/ddiff, -dE/R_i = dE/ddiff
                i_forces = torch.zeros((forces_dim, 3), device=edge_vectors.device, dtype=torch.float32)
                j_forces = torch.zeros_like(i_forces)
                i_forces.index_add_(0, edge_indices[:, 0], dE_ddiff)
                j_forces.index_add_(0, edge_indices[:, 1], -dE_ddiff)
                forces = i_forces + j_forces
                data = replace_properties(data, forces=forces)
                    
        return data