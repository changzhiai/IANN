from .mace import MACE
from .painn import PaiNN
from .nequip import NequIP
from .equiformerV2 import EquiformerV2

__all__ = [
    "MACE",
    "PaiNN", 
    "NequIP",
    "EquiformerV2",
]

# Model registry for easy access
MODEL_REGISTRY = {
    "mace": MACE,
    "painn": PaiNN,
    "nequip": NequIP,
    "equiformerV2": EquiformerV2,
}
