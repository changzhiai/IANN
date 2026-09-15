"""
Foundation models module for IANN.

This module provides easy access to pre-trained foundation models. A released
model is requested by name and downloaded from the HuggingFace Hub on first use;
see :func:`iann.foundations.foundation_models.get_foundation_model_path`.
"""

from iann.foundations.foundation_models import (
    FoundationModelDownloadError,
    UnknownFoundationModelError,
    download_foundation_model,
    foundation_model,
    foundation_model_info,
    foundation_models_table,
    get_foundation_model_path,
    is_cached,
    list_available_models,
    list_foundation_models,
    load_foundation_model,
)
from iann.foundations.registry import FOUNDATION_MODELS, HF_REPO_ID, HF_REVISION

__all__ = [
    # Original public surface, kept first for stability.
    "foundation_model",
    "list_available_models",
    # Resolution and metadata.
    "get_foundation_model_path",
    "load_foundation_model",
    "download_foundation_model",
    "is_cached",
    "foundation_model_info",
    "list_foundation_models",
    "foundation_models_table",
    # Catalog.
    "FOUNDATION_MODELS",
    "HF_REPO_ID",
    "HF_REVISION",
    # Errors.
    "UnknownFoundationModelError",
    "FoundationModelDownloadError",
]
