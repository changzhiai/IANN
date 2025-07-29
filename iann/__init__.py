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
def __getattr__(name):
    """Lazy load model classes when accessed as attributes"""
    if name in ["PaiNN", "NequIP", "MACE", "EquiformerV2"]:
        return get_model_class(name.lower())
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
