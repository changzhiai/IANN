# Lazy loading functions for models
def _load_mace():
    """Lazy load MACE model"""
    from .mace import MACE
    return MACE

def _load_painn():
    """Lazy load PaiNN model"""
    from .painn import PaiNN
    return PaiNN

def _load_nequip():
    """Lazy load NequIP model"""
    from .nequip import NequIP
    return NequIP

def _load_equiformerV2():
    """Lazy load EquiformerV2 model"""
    from .equiformerV2 import EquiformerV2
    return EquiformerV2

def _load_hotmem():
    """Lazy load HOTMEM model"""
    from .hotmem import HOTMEM
    return HOTMEM

__all__ = [
    "MACE",
    "PaiNN", 
    "NequIP",
    "EquiformerV2",
    "HOTMEM",
    "get_model_class",
]

# Model registry with lazy loading for easy access
MODEL_REGISTRY = {
    "mace": _load_mace,
    "painn": _load_painn,
    "nequip": _load_nequip,
    "equiformerV2": _load_equiformerV2,
    "hotmem": _load_hotmem,
}

# Convenience function for direct access (lazy loaded)
def get_model_class(model_name):
    """Get a model class by name with lazy loading"""
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]()

# For backward compatibility, provide direct access to model classes
# These will only be imported when actually accessed
def __getattr__(name):
    """Lazy load model classes when accessed as attributes"""
    if name == "MACE":
        return _load_mace()
    elif name == "PaiNN":
        return _load_painn()
    elif name == "NequIP":
        return _load_nequip()
    elif name == "EquiformerV2":
        return _load_equiformerV2()
    elif name == "HOTMEM":
        return _load_hotmem()
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
