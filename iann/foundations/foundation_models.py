"""
Foundation models module for IANN.

This module provides easy access to pre-trained foundation models.
"""

import os
import torch

def get_foundation_model_path(model_name="painn_oc.pt"):
    """
    Get the absolute path to a foundation model file.
    
    Parameters
    ----------
    model_name : str, optional
        Name of the foundation model file (default: "painn_oc.pt")
    
    Returns
    -------
    str
        Absolute path to the foundation model file
    
    Examples
    --------
    >>> from iann.foundations.foundation_models import get_foundation_model_path
    >>> path = get_foundation_model_path("painn_oc.pt")
    """
    # Get the directory of this file
    _here = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(_here, model_name)

def load_foundation_model(model_name="painn_oc.pt", device=None):
    """
    Load a foundation model checkpoint.
    
    Parameters
    ----------
    model_name : str, optional
        Name of the foundation model file (default: "painn_oc.pt")
    device : str or torch.device, optional
        Device to load the model on (default: 'cpu' or 'cuda' if available)
    
    Returns
    -------
    dict
        State dictionary containing the model checkpoint
    
    Examples
    --------
    >>> from iann.foundations.foundation_models import load_foundation_model
    >>> state_dict = load_foundation_model("painn_oc.pt")
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_path = get_foundation_model_path(model_name)
    state_dict = torch.load(model_path, map_location=device)
    return state_dict

def foundation_model(model_name="painn_oc.pt"):
    """
    Get the path to a foundation model.
    
    This is a convenience function that automatically resolves the path
    to foundation models regardless of where the script is run from.
    
    Parameters
    ----------
    model_name : str, optional
        Name of the foundation model file (default: "painn_oc.pt")
    
    Returns
    -------
    str
        Absolute path to the foundation model file
    
    Examples
    --------
    >>> from iann.foundations import foundation_model
    >>> from iann.calculators import MLCalculator
    >>> 
    >>> # Use directly
    >>> calc = MLCalculator(foundation_model("painn_oc.pt"))
    >>> 
    >>> # Or with default
    >>> calc = MLCalculator(foundation_model())
    """
    return get_foundation_model_path(model_name)

def list_available_models():
    """
    List all available foundation models.
    
    Returns
    -------
    list
        List of available foundation model filenames
    
    Examples
    --------
    >>> from iann.foundations import list_available_models
    >>> models = list_available_models()
    >>> print(models)
    """
    _here = os.path.abspath(os.path.dirname(__file__))
    models = []
    for file in os.listdir(_here):
        if file.endswith('.pt'):
            models.append(file)
    return sorted(models)
