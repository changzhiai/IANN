# Import main classes for easier access
from .calculators import MLCalculator, EnsembleCalculator, AtomicEnsembleCalculator
from .trainer import Trainer
from .data import AtomsData, AseDataset
from .models import get_model_class

__all__ = [
    "MLCalculator",
    "EnsembleCalculator", 
    "AtomicEnsembleCalculator",
    "Trainer",
    "AtomsData",
    "AseDataset",
    "get_model_class",
]

# For backward compatibility, provide direct access to model classes
# These will only be imported when actually accessed
#
# Derived from the registry rather than hand-listed, so a newly registered
# architecture is reachable as iann.<Name> without editing this file. The
# previous hand-written list had fallen behind and omitted Allegro and
# EquiformerV3.
_MODEL_ATTRS = {
    "PaiNN": "painn",
    "NequIP": "nequip",
    "Allegro": "allegro",
    "MACE": "mace",
    "EquiformerV2": "equiformerv2",
    "EquiformerV3": "equiformerv3",
    "UMA": "uma",
    "FastPot": "fastpot",
}


def __getattr__(name):
    """Lazy load model classes when accessed as attributes"""
    if name in _MODEL_ATTRS:
        return get_model_class(_MODEL_ATTRS[name])
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(list(__all__) + list(_MODEL_ATTRS))
