import torch
from torch import nn
from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.o3 import Linear
import abc, math
from ase.data import atomic_numbers
import warnings
from e3nn.o3 import Linear, TensorProduct, FullyConnectedTensorProduct
from e3nn.nn import FullyConnectedNet
from e3nn.nn import Gate, NormActivation
from typing import Dict, List, Optional, Union, Callable
from iann.data.data import AtomsData

class Transform(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__()
    
    @abc.abstractmethod
    def forward(self):
        raise NotImplementedError
    
class UnitTransform(Transform):
    def __init__(
        self,
        unit_dict: Dict[str, float]
    ) -> None:
        super().__init__()
        
        self.unit_dict = unit_dict
    
    def forward(self, data):
        for k, v in self.unit_dict.items():
            setattr(data, k, getattr(data, k) * v)
        
        return data   

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
        # Handle dictionary inputs for TorchScript compatibility
        is_dict = isinstance(data, dict)
        
        if (is_dict and 'atomic_types' in data) or (not is_dict and hasattr(data, 'atomic_types')):
            warnings.warn("Data already contains mapped types. This will be overwrited.")
        
        types = self.transform(data['atomic_numbers'] if is_dict else data.atomic_numbers)
        
        if is_dict:
            data['atomic_types'] = types
        else:
            data.atomic_types = types
            
        assert torch.all(types >= 0), "Provided data contains species not defined in TypeMapper!"
        return data
        
    def transform(self, numbers: torch.Tensor) -> torch.Tensor:
        if numbers.max() > 119 or numbers.min() < 1:
            raise ValueError("Provided atomic numbers are not in the periodic table!")
        types = self.Z_to_index[numbers]
        return types
    
    def untransform(self, types: torch.Tensor) -> torch.Tensor:
        return self.index_to_Z[types]

@compile_mode("script")
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
            
    def forward(self, data):
        # Handle dictionary inputs for TorchScript compatibility
        is_dict = isinstance(data, dict)
        
        if (is_dict and 'atomic_types' not in data) or (not is_dict and not hasattr(data, 'atomic_types')):
            if self.type_mapper is not None:
                data = self.type_mapper(data)
            else:
                if is_dict:
                    data['atomic_types'] = data['atomic_numbers'] - 1
                else:
                    data.atomic_types = data.atomic_numbers - 1
                    
        onehot = torch.nn.functional.one_hot(
            data['atomic_types'] if is_dict else data.atomic_types, 
            num_classes=self.num_elements
        ).to(
            device=data['positions'].device if is_dict else data.positions.device, 
            dtype=data['positions'].dtype if is_dict else data.positions.dtype
        )
        
        if is_dict:
            data['node_attr'] = onehot
        else:
            data.node_attr = onehot
            
        if self.set_features:
            if is_dict:
                data['node_feat'] = onehot
            else:
                data.node_feat = onehot
                
        return data

class AtomwiseLinear(torch.nn.Module):
    def __init__(
        self,
        irreps_in: Optional[o3.Irreps]=None,
        irreps_out: Optional[o3.Irreps]=None,
        field: str='node_feat',
        out_field: Optional[str]=None,
    ):
        super().__init__()
        self.irreps_in: Optional[o3.Irreps] = irreps_in
        if irreps_out is None:
            irreps_out = irreps_in
        self.irreps_out = irreps_out
        
        self.linear = Linear(
            irreps_in=self.irreps_in, irreps_out=self.irreps_out
        )
        self.field = field
        self.out_field = out_field if out_field is not None else self.field

    def forward(self, data):
        # Handle dictionary inputs for TorchScript compatibility
        is_dict = isinstance(data, dict)
        
        if is_dict:
            input_tensor = data[self.field]
            data[self.out_field] = self.linear(input_tensor)
        else:
            setattr(data, self.out_field, self.linear(getattr(data, self.field)))
            
        return data

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
        self.irreps_out = o3.Irreps([(num_basis, (0, 1))])

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

@torch.jit.script
def _poly_cutoff(x: torch.Tensor, factor: float, p: float = 6.0) -> torch.Tensor:
    x = x * factor

    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
    out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))

    return out * (x < 1.0)

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
    
@compile_mode("script")
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

    def forward(self, data):
        # Handle dictionary inputs for TorchScript compatibility
        is_dict = isinstance(data, dict)
        
        edge_diff = data['edge_vectors'] if is_dict else data.edge_vectors
        edge_dist = torch.linalg.norm(edge_diff, dim=1)
        embedding = self.basis(edge_dist) * self.cutoff_fn(edge_dist)[:, None]
        
        if is_dict:
            data['edge_dist_embedding'] = embedding
        else:
            data.edge_dist_embedding = embedding
        
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

    def forward(self, data):
        # Handle dictionary inputs for TorchScript compatibility
        is_dict = isinstance(data, dict)
        
        edge_vectors = data['edge_vectors'] if is_dict else data.edge_vectors
        embedding = self.sh(edge_vectors)
        
        if is_dict:
            data['edge_diff_embedding'] = embedding
        else:
            data.edge_diff_embedding = embedding
            
        return data

@torch.jit.script
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
    use_sc: bool

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
        self.linear_1 = Linear(
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
                        irreps_mid.append((mul, ir_out))
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

        tp = TensorProduct(
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

        self.linear_2 = Linear(
            # irreps_mid has uncoallesed irreps because of the uvu instructions,
            # but there's no reason to treat them seperately for the Linear
            # Note that normalization of o3.Linear changes if irreps are coallesed
            # (likely for the better)
            irreps_in=irreps_mid.simplify(),
            irreps_out=feature_irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        self.sc = None
        if self.use_sc:
            self.sc = FullyConnectedTensorProduct(
                feature_irreps_in,
                irreps_in['node_attr'],
                feature_irreps_out,
            )

    def forward(self, data):
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
        # Handle dictionary inputs for TorchScript compatibility
        is_dict = isinstance(data, dict)
        
        weight = self.fc(data['edge_dist_embedding'] if is_dict else data.edge_dist_embedding)
        x = data['node_feat'] if is_dict else data.node_feat
        edge_idx = data['edge_indices'] if is_dict else data.edge_indices  # i, j index
        node_attr = data['node_attr'] if is_dict else data.node_attr
        edge_diff_embedding = data['edge_diff_embedding'] if is_dict else data.edge_diff_embedding

        if self.sc is not None:
            sc = self.sc(x, node_attr)

        x = self.linear_1(x)
        edge_features = self.tp(
            x[edge_idx[:, 1]], edge_diff_embedding, weight
        )
        x = scatter_add(edge_features, edge_idx[:, 0], dim_size=len(x), dim=0)

        # Necessary to get TorchScript to be able to type infer when its not None
        # avg_num_neigh: Optional[float] = self.avg_num_neighbors
        # if avg_num_neigh is not None:
        x = x.div(self.avg_num_neighbors**0.5)

        x = self.linear_2(x)

        if self.sc is not None:
            x = x + sc

        if is_dict:
            data['node_feat'] = x
        else:
            data.node_feat = x
        
        return data
    
    def datamodule(self, _datamodule):
        if not self._initialized:
            avg_num_neigh = _datamodule._get_avg_num_neighbors()
            if avg_num_neigh is not None:
                self.avg_num_neighbors = torch.tensor([avg_num_neigh])


#@torch.jit.script
def ShiftedSoftPlus(x):
    return torch.nn.functional.softplus(x) - math.log(2.0)

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

    resnet: bool

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
                (mul, ir)
                for mul, ir in self.feature_irreps_hidden
                if ir.l == 0
                and tp_path_exists(irreps_layer_out_prev, edge_diff_irreps, ir)
            ]
        )

        irreps_gated = o3.Irreps(
            [
                (mul, ir)
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
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

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

        # # TODO: last convolution should go to explicit irreps out
        # logging.debug(
        #     f" parameters used to initialize {convolution.__name__}={convolution_kwargs}"
        # )

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

    def forward(self, data):
        # save old features for resnet
        # Handle dictionary inputs for TorchScript compatibility
        is_dict = isinstance(data, dict)

        old_node_feat = data['node_feat'] if is_dict else data.node_feat
        # run convolution
        data = self.conv(data)
        # do nonlinearity
        if is_dict:
            data['node_feat'] = self.equivariant_nonlin(data['node_feat'])
        else:
            data.node_feat = self.equivariant_nonlin(data.node_feat)

        if self.resnet:
            if is_dict:
                data['node_feat'] += old_node_feat
            else:
                data.node_feat += old_node_feat

        return data

class NequIP(torch.nn.Module):
    def __init__(
        self,
        cutoff: float,
        num_interactions: int,
        species: Optional[List[str]] = None,
        num_elements: Optional[int] = None,
        hidden_irreps: Union[o3.Irreps, str, None] = None,
        edge_sh_irreps: Union[o3.Irreps, str, None] = None,
        node_irreps: Union[o3.Irreps, str, None] = None,
        MLP_irreps: Union[o3.Irreps, str, None] = None,
        lmax: int = 2,
        parity: bool = True,
        num_features: Optional[int] = None,
        num_basis: int = 8,
        power: int = 6,
        # parameters for interaction blocks and convnet
        resnet: bool = False,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "ssp", "o": "abs"},
        convolution_kwargs: dict = {},
        **kwargs,
    ) -> None:
        """
        Args:
            cutoff (float): Cutoff radius
            num_interactions (int): Number of interaction blocks
            species (List[str]): List of species
            num_elements (Optional[int], optional): Number of elements. Defaults to None.
            hidden_irreps (Union[o3.Irreps, str, None], optional): Hidden irreps. Defaults to None.
            edge_sh_irreps (Union[o3.Irreps, str, None], optional): Edge irreps. Defaults to None.
            node_irreps (Union[o3.Irreps, str, None], optional): Node irreps. Defaults to None.
            MLP_irreps (Union[o3.Irreps, str, None], optional): MLP irreps. Defaults to None.
            lmax (int, optional): Maximum l value for spherical harmonics. Defaults to 2.
            parity (bool, optional): Parity. Defaults to True.
            num_features (Optional[int], optional): Number of features. Defaults to None.
            num_basis (int, optional): Number of basis. Defaults to 8.
            power (int, optional): Power of radial basis. Defaults to 6.
            resnet (bool, optional): ResNet. Defaults to False.
            nonlinearity_type (str, optional): Type of nonlinearity. Defaults to "gate".
            nonlinearity_scalars (Dict[int, Callable], optional): Nonlinearity for scalars. Defaults to {"e": "ssp", "o": "tanh"}.
            nonlinearity_gates (Dict[int, Callable], optional): Nonlinearity for gates. Defaults to {"e": "ssp", "o": "abs"}.
            convolution_kwargs (dict, optional): Convolution kwargs. Defaults to {}.
        """
        super().__init__()
        self.cutoff = cutoff
        self.num_features = num_features
        self.lmax = lmax
        self.parity = parity
        
        if num_elements is None:
            num_elements = len(species) if species is not None else 119
        
        ## handling irreps
        # chemical embedding irreps
        if node_irreps is None:
            self.node_irreps = o3.Irreps([(num_features, (0, 1))])
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
        # hidden feature irreps
        if hidden_irreps is None:
            self.hidden_irreps = o3.Irreps(
                [
                    (num_features, (l, p))
                    for p in ((1, -1) if parity else (1,))
                    for l in range(lmax + 1)
                ]
            )
        elif isinstance(hidden_irreps, str):
            self.hidden_irreps = o3.Irreps(hidden_irreps)
        else:
            self.hidden_irreps = hidden_irreps
        # MLP_irreps
        if MLP_irreps is None:
            self.MLP_irreps = o3.Irreps([(max(1, num_features // 2), (0, 1))])
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
            irreps_in=self.irreps_in['node_attr'], # from OneHotAtomEncoding
            irreps_out=self.node_irreps,
        )
        self.irreps_in['node_feat'] = self.embeddings.chemical_embedding.irreps_out
        
        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            interaction = InteractionLayer(
                irreps_in=self.irreps_in, 
                feature_irreps_hidden=self.hidden_irreps,
                convolution_kwargs=convolution_kwargs,
                resnet=resnet,
                nonlinearity_type=nonlinearity_type,
                nonlinearity_scalars=nonlinearity_scalars,
                nonlinearity_gates=nonlinearity_gates,
            )
            self.interactions.append(interaction)
            self.irreps_in.update(interaction.irreps_out)
        
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
        self.atomwise_reduce = AtomwiseReduce(output_key='energy')
        
        self.compute_forces = False
        if 'compute_forces' in kwargs.keys():
            if kwargs['compute_forces']:
                self.compute_forces = True
                self.gradient_output = GradientOutput(model_outputs=['forces'])

    def forward(self, data: AtomsData) -> dict:
        """
        Args:
            data (AtomsData): A NamedTuple of model inputs

        Returns:
            dict: A dictionary with keys 'energy' and 'forces'
        """
        # Handle both dictionary and object inputs for TorchScript compatibility
        if isinstance(data, dict):
            # Dictionary access
            species = data['atomic_numbers']
            pos = data['positions'] 
            edge_index = data['edge_indices']
            edge_cell_shift = data['edge_shifts'] if 'edge_shifts' in data else None
            edge_vec = data['edge_vectors']
            cell = data['cell'] if 'cell' in data else None

            # Determine edge mask
            if 'edge_mask' in data:
                mask = data['edge_mask'].bool()
            else:
                mask = torch.ones(edge_index.shape[1], device=edge_index.device, dtype=torch.bool)

            # Create a new dictionary to store data
            processed_data = dict(data)
        else:
            # Object access for AtomsData or ScriptableAtomsBatch
            species = data.atomic_numbers
            pos = data.positions
            edge_index = data.edge_indices
            edge_cell_shift = data.edge_shifts if hasattr(data, "edge_shifts") else None
            edge_vec = data.edge_vectors
            cell = data.cell if hasattr(data, "cell") else None

            # Determine edge mask
            if hasattr(data, "edge_mask"):
                mask = data.edge_mask.bool()
            else:
                mask = torch.ones(edge_index.shape[1], device=edge_index.device, dtype=torch.bool)

            # Convert to dictionary for easier processing
            processed_data = {}
            # Copy existing attributes
            for key in dir(data):
                if not key.startswith('_') and hasattr(data, key) and not callable(getattr(data, key)):
                    processed_data[key] = getattr(data, key)

        for m in self.embeddings.values():
            processed_data = m(processed_data)
            
        for m in self.interactions:
            processed_data = m(processed_data)
        
        processed_data['atomic_energy'] = self.readout_mlp(processed_data['node_feat']).squeeze()
        processed_data = self.atomwise_reduce(processed_data)

        if self.compute_forces:
            processed_data = self.gradient_output(processed_data)

        return processed_data
    
class AtomwiseReduce(nn.Module):
    def __init__(
        self,
        output_key: str = "energy",
        per_atom_output: bool = False,
        aggregation_mode: str = "sum",     # should be sum or mean
    ) -> None:
        super().__init__()
        self.model_outputs = [output_key]
        if per_atom_output:
            self.model_outputs.append(output_key + '_per_atom')
        self.aggregation_mode = aggregation_mode
        self.output_key = output_key
        self.per_atom_output = per_atom_output
    
    def forward(self, data):
        # Handle dictionary inputs for TorchScript compatibility
        is_dict = isinstance(data, dict)
        
        if is_dict:
            num_atoms = data['num_atoms']
            image_idx = data['image_idx']
            atomic_energy = data['atomic_energy']
            dtype = data['edge_vectors'].dtype
        else:
            num_atoms = data.num_atoms
            image_idx = data.image_idx
            atomic_energy = data.atomic_energy
            dtype = data.edge_vectors.dtype
            
        y = torch.zeros_like(
            num_atoms, 
            dtype=dtype
        )  
        y.index_add_(0, image_idx, atomic_energy)
        
        if self.aggregation_mode == "mean":
            y = y / num_atoms
        
        if is_dict:
            data[self.output_key] = y
        else:
            setattr(data, self.output_key, y)
            
        if self.per_atom_output:
            if is_dict:
                data[self.output_key + '_per_atom'] = atomic_energy
            else:
                setattr(data, self.output_key + '_per_atom', atomic_energy)
        
        return data
    
class GradientOutput(torch.nn.Module):
    def __init__(
        self,
        grad_on_edge_diff: bool = True,
        grad_on_positions: bool = False,
        model_outputs: List[str] = ['forces'],       # properties that need to be calculated, can be forces, stress, virial, etc.
        update_callback: Optional[Callable] = None,  # Add a callback parameter
    ) -> None:
        # TODO: define a set for allowed model outputs
        super().__init__()
        self.grad_on_edge_diff = grad_on_edge_diff
        self.grad_on_positions = grad_on_positions
        self.update_callback = update_callback
        self.model_outputs = model_outputs

    @torch.jit.ignore
    def update_model_outputs(self, outputs: Union[List[str], str]):
        if isinstance(outputs, str):
            self.model_outputs.append(outputs)
        else:
            self.model_outputs.extend(outputs)
        # update parent model
        if self.update_callback:
            self.update_callback()

    def forward(self, data, training: bool=True,):
        # Handle dictionary inputs for TorchScript compatibility
        is_dict = isinstance(data, dict)
        
        # Helper functions to get attributes
        def get_attr(key):
            return data[key] if is_dict else getattr(data, key)
        
        def set_attr(key, value):
            if is_dict:
                data[key] = value
            else:
                setattr(data, key, value)
        
        # Get necessary attributes
        energy = get_attr('energy')
        
        if self.grad_on_edge_diff:
            edge_diff = get_attr('edge_vectors')
            forces_dim = int(torch.sum(get_attr('num_atoms')))
            edge_idx = get_attr('edge_indices')
            if 'forces' in self.model_outputs:
                grad_outputs : List[Optional[torch.Tensor]] = [torch.ones_like(energy)]    # for model deploy
                dE_ddiff = torch.autograd.grad(
                    [energy,],
                    [edge_diff,],
                    grad_outputs=grad_outputs,
                    retain_graph=training,
                    create_graph=training,
                )
                dE_ddiff = torch.zeros_like(get_attr('positions')) if dE_ddiff is None else dE_ddiff[0]   # for torch.jit.script
                assert dE_ddiff is not None
                
                # diff = R_j - R_i, so -dE/dR_j = -dE/ddiff, -dE/R_i = dE/ddiff
                i_forces = torch.zeros((forces_dim, 3), device=edge_diff.device, dtype=edge_diff.dtype)
                j_forces = torch.zeros_like(i_forces)
                i_forces.index_add_(0, edge_idx[:, 0], dE_ddiff)
                j_forces.index_add_(0, edge_idx[:, 1], -dE_ddiff)
                forces = i_forces + j_forces
                set_attr('forces', forces)

                # Reference: https://en.wikipedia.org/wiki/Virial_stress
                # This method calculates virials by giving pair-wise force components
                
                if 'stress' in self.model_outputs or 'virial' in self.model_outputs:
                    image_idx = get_attr('image_idx')
                    atomic_virial = torch.einsum("ij, ik -> ijk", edge_diff, dE_ddiff)           # I'm quite not sure if a negative sign should be added before dE_ddiff, but I think it should be right
                    # stress = torch.zeros_like(cell).index_add(0, , atomic_stress)
                    atomic_virial = torch.zeros(
                        (forces_dim, 3, 3),                                         
                        dtype=forces.dtype,
                        # it seens like the calculation is not very right... because f_ij is not absolutely right here. Maybe we need to do something like in force calculation
                        # add i_stress and j_stress together then it is the total stress. need verification
                        device=forces.device).index_add(0, edge_idx[:, 0], atomic_virial)
                    # j_stress = torch.zeros_like(i_stress).index_add(0, edge_idx[:, 1], -atomic_stress)
                    # atomic_stress = i_stress + j_stress          
                    virial = torch.zeros(
                        energy.shape[0], 3, 3, 
                        dtype=forces.dtype, 
                        device=forces.device).index_add(0, image_idx, atomic_virial)  # don't need to divide by two
                    set_attr('virial', virial.view(-1, 9)[:, [0, 4, 8, 5, 2, 1]])
                    has_cell = 'cell' in data if is_dict else hasattr(data, "cell")
                    if has_cell and 'stress' in self.model_outputs:
                        cell = get_attr('cell').view(-1, 3, 3)
                        volumes = torch.sum(cell[:, 0] * cell[:, 1].cross(cell[:, 2], dim=-1), dim=1)
                        stress = - virial / volumes[:, None, None]
                        set_attr('stress', stress.view(-1, 9)[:, [0, 4, 8, 5, 2, 1]])
            
        elif self.grad_on_positions:
            grad_outputs : List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
            if 'forces' in self.model_outputs:
                positions = get_attr('positions')
                grad_inputs = [positions]
                if 'stress' in self.model_outputs and ('strain' in data if is_dict else hasattr(data, 'strain')):
                    grad_inputs.append(get_attr('strain'))
                grads = torch.autograd.grad(
                    [energy,],
                    grad_inputs,
                    grad_outputs=grad_outputs,
                    retain_graph=training,
                    create_graph=training,
                )
                dEdR = grads[0]
                if dEdR is None:
                    dEdR = torch.zeros_like(positions)
                set_attr('forces', -dEdR)
                    
                if 'stress' in self.model_outputs:
                    has_cell = 'cell' in data if is_dict else hasattr(data, "cell")
                    if has_cell:
                        stress = grads[1]
                        if stress is None:
                            stress = torch.zeros_like(get_attr('cell'))
                        cell = get_attr('cell').view(-1, 3, 3)
                        volumes = torch.sum(cell[:, 0] * cell[:, 1].cross(cell[:, 2], dim=-1), dim=1)
                        stress /= volumes[:, None, None]
                        set_attr('stress', stress.view(-1, 9)[:, [0, 4, 8, 5, 2, 1]])
        
        else:
            raise ValueError("Gradients must be calculated with respect to positions or R_ij. Nothing is given!")
                    
        return data