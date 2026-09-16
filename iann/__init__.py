"""IANN: an equivariant interatomic neural network framework.

Every public name is resolved lazily. ``import iann`` therefore costs almost
nothing and, importantly, does not require the numerical stack to be working:
``iann.agent``'s diagnostics have to be reachable in an environment where torch
or asap3 is broken, which is exactly when they are wanted. Previously this
module imported the calculators, trainer and data layer eagerly, so a broken
asap3 made even ``iann doctor`` impossible to run.

Access is unchanged -- ``import iann; iann.MLCalculator`` and
``from iann import Trainer`` both still work, and raise the underlying
ImportError at first use rather than at import time.
"""

# name -> (submodule, attribute); resolved on first access by __getattr__ below.
_LAZY_ATTRS = {
    "MLCalculator": ("iann.calculators", "MLCalculator"),
    "EnsembleCalculator": ("iann.calculators", "EnsembleCalculator"),
    "AtomicEnsembleCalculator": ("iann.calculators", "AtomicEnsembleCalculator"),
    "Trainer": ("iann.trainer", "Trainer"),
    "AtomsData": ("iann.data", "AtomsData"),
    "AseDataset": ("iann.data", "AseDataset"),
    "get_model_class": ("iann.models", "get_model_class"),
}

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
    """Resolve model classes and the top-level API on first access."""
    if name in _LAZY_ATTRS:
        module_name, attr = _LAZY_ATTRS[name]
        import importlib
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value          # cache, so later access is a plain lookup
        return value
    if name in _MODEL_ATTRS:
        from .models import get_model_class
        return get_model_class(_MODEL_ATTRS[name])
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(list(__all__) + list(_MODEL_ATTRS))
