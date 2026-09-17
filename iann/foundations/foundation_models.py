"""
Foundation models module for IANN.

Resolves a foundation model to a local checkpoint path. A name is looked up in
this order:

1. an explicit path to an existing ``.pt`` file, returned as-is;
2. a checkpoint bundled inside this package (``painn_oc.pt`` and friends);
3. an entry in the catalog of released models (:mod:`iann.foundations.registry`),
   served from the local cache if present and downloaded from the HuggingFace Hub
   otherwise.

The first download of a catalog model needs network access; afterwards it is
served from the cache. See :func:`download_foundation_model` for prefetching on a
login node, and ``HF_HOME`` for placing the cache on shared scratch.
"""

import difflib
import os
import warnings
from typing import List, Optional

import torch

from iann.foundations import registry
from iann.foundations.registry import (
    FOUNDATION_MODELS,
    HF_REPO_ID,
    HF_REPO_TYPE,
    HF_REVISION,
    normalize_name,
    resolve_entry,
)

_HERE = os.path.abspath(os.path.dirname(__file__))

#: Environment variable overriding where downloaded checkpoints are cached.
CACHE_ENV_VAR = "IANN_FOUNDATION_CACHE"


class UnknownFoundationModelError(ValueError):
    """Raised when a name is neither a readable path nor a known model."""


class FoundationModelDownloadError(RuntimeError):
    """Raised when a known model could not be fetched from the Hub."""


def _looks_like_path(name: str) -> bool:
    return os.sep in str(name) or "/" in str(name) or str(name).endswith(".pt")


def _bundled_path(model_name: str) -> Optional[str]:
    """Path to a checkpoint shipped inside the package, if it is present."""
    candidate = os.path.join(_HERE, os.path.basename(str(model_name)))
    return candidate if os.path.isfile(candidate) else None


def _grouped_name_list() -> str:
    """The catalog, grouped by functional, for use in error messages."""
    lines = []
    for functional in ("PBE", "RPBE", "r2SCAN"):
        names = [
            e.name
            for e in FOUNDATION_MODELS.values()
            if e.group == "painn" and e.functional == functional
        ]
        if names:
            lines.append("  %-8s %s" % (functional, "  ".join(names)))
    arch = [e.name for e in FOUNDATION_MODELS.values() if e.group == "arch"]
    if arch:
        lines.append("  %-8s %s" % ("arch", "  ".join(arch)))
    bundled = [f for f in registry.BUNDLED_FILENAMES if _bundled_path(f)]
    if bundled:
        lines.append("  %-8s %s" % ("bundled", "  ".join(bundled)))
    return "\n".join(lines)


def _unknown_model_error(model_name: str) -> UnknownFoundationModelError:
    if _looks_like_path(model_name):
        resolved = os.path.abspath(os.path.expanduser(str(model_name)))
        return UnknownFoundationModelError(
            "%r looks like a file path, but no such file exists (resolved to %s). "
            "If you meant a released foundation model, see "
            "iann.foundations.list_foundation_models()." % (model_name, resolved)
        )
    suggestions = difflib.get_close_matches(
        normalize_name(model_name), list(FOUNDATION_MODELS), n=3, cutoff=0.6
    )
    msg = "unknown foundation model %r." % model_name
    if suggestions:
        msg += "\n\nDid you mean: %s?" % ", ".join(suggestions)
    msg += (
        "\n\nAvailable (downloaded on first use from %s):\n%s"
        "\n\nTo use your own checkpoint, pass a path to an existing .pt file instead."
        % (HF_REPO_ID, _grouped_name_list())
    )
    return UnknownFoundationModelError(msg)


def _hf_download(entry, *, cache_dir, revision, force_download, local_files_only, token):
    """Fetch one catalog entry through huggingface_hub.

    Imported lazily so that resolving bundled checkpoints and explicit paths does
    not require the dependency at all.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise FoundationModelDownloadError(
            "downloading foundation model %r requires the 'huggingface_hub' "
            "package, which is an optional dependency and is not installed. "
            "Install it with\n"
            "    pip install \"iann[foundations]\"\n"
            "or pass a path to a checkpoint you already have." % entry.name
        ) from exc

    try:
        from huggingface_hub.errors import (
            EntryNotFoundError,
            GatedRepoError,
            LocalEntryNotFoundError,
            RepositoryNotFoundError,
        )
    except ImportError:  # older huggingface_hub exposed these at the top level
        from huggingface_hub.utils import (  # type: ignore
            EntryNotFoundError,
            GatedRepoError,
            LocalEntryNotFoundError,
            RepositoryNotFoundError,
        )

    kwargs = dict(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        filename=entry.path,
        revision=revision or HF_REVISION,
        cache_dir=cache_dir or os.environ.get(CACHE_ENV_VAR) or None,
        local_files_only=local_files_only,
        force_download=force_download,
        library_name="iann",
    )
    if token is not None:
        kwargs["token"] = token

    try:
        return hf_hub_download(**kwargs)
    except LocalEntryNotFoundError:
        if local_files_only:
            raise FoundationModelDownloadError(
                "foundation model %r is not in the local cache and no download was "
                "attempted (local_files_only=True). Prefetch it with\n"
                "    hf download %s %s" % (entry.name, HF_REPO_ID, entry.path)
            )
        raise FoundationModelDownloadError(
            "could not obtain foundation model %r (%s :: %s @ %s): it is not cached "
            "and the Hub is unreachable.\n\nThe first use of a foundation model needs "
            "network access. To prefetch on a login node and run offline afterwards:\n"
            "    hf download %s %s\nOr set HF_HOME to a shared scratch directory so "
            "one cache serves every node."
            % (entry.name, HF_REPO_ID, entry.path, revision or HF_REVISION,
               HF_REPO_ID, entry.path)
        )
    except (RepositoryNotFoundError, GatedRepoError) as exc:
        raise FoundationModelDownloadError(
            "could not obtain foundation model %r: the repository %s is private, "
            "gated, or does not exist. If you have access, run `hf auth login` or "
            "set HF_TOKEN, then retry. (%s)" % (entry.name, HF_REPO_ID, exc)
        ) from exc
    except EntryNotFoundError as exc:
        raise FoundationModelDownloadError(
            "the repository %s is reachable but does not contain %r, which the IANN "
            "catalog lists for model %r. The catalog and the repository are out of "
            "step; please report this. (%s)"
            % (HF_REPO_ID, entry.path, entry.name, exc)
        ) from exc
    except Exception as exc:  # network errors, proxies, offline mode
        raise FoundationModelDownloadError(
            "could not obtain foundation model %r from %s: %s"
            % (entry.name, HF_REPO_ID, exc)
        ) from exc


def get_foundation_model_path(
    model_name: str = "painn_oc.pt",
    *,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token=None,
) -> str:
    """
    Resolve a foundation model to an absolute path to a ``.pt`` checkpoint.

    Parameters
    ----------
    model_name : str, optional
        A released model name (``"rpbe-all"``), a bundled filename
        (``"painn_oc.pt"``), or a path to a checkpoint of your own.
    cache_dir : str, optional
        Where to cache downloads. Defaults to ``$IANN_FOUNDATION_CACHE``, then to
        the HuggingFace cache (``$HF_HOME``).
    revision : str, optional
        Hub revision to fetch; defaults to the pinned catalog revision.
    force_download : bool, optional
        Re-download even if a cached copy exists, e.g. if it is corrupt.
    local_files_only : bool, optional
        Never touch the network; raise if the model is not already local.
    token : str or bool, optional
        Hub token, for a private or gated repository.

    Returns
    -------
    str
        Absolute path to the checkpoint.

    Raises
    ------
    UnknownFoundationModelError
        The name is neither a readable path nor a catalog entry. Raised before
        any I/O.
    FoundationModelDownloadError
        A catalog model could not be fetched.

    Examples
    --------
    >>> from iann.foundations import foundation_model
    >>> path = foundation_model("rpbe-all")          # downloads on first use
    >>> path = foundation_model("painn_oc.pt")       # bundled with the package
    >>> path = foundation_model("output/model.pt")   # your own checkpoint
    """
    if not isinstance(model_name, (str, os.PathLike)):
        raise TypeError(
            "model_name must be a string or path, got %r" % type(model_name).__name__
        )
    name = str(model_name)

    # 1. An explicit path to a file that exists. isfile(), not exists(), so that a
    #    directory sharing a name with a catalog alias cannot shadow it.
    explicit = os.path.abspath(os.path.expanduser(name))
    if not force_download and os.path.isfile(explicit):
        return explicit

    # 2. A checkpoint bundled inside the package. Preserves the historical
    #    behaviour of this function exactly.
    if not force_download:
        bundled = _bundled_path(name)
        if bundled is not None:
            return bundled

    # 3. The catalog. A miss raises before any I/O.
    entry = resolve_entry(name)
    if entry is None:
        raise _unknown_model_error(name)

    # 4./5. Cache, then download.
    if not force_download:
        try:
            path = _hf_download(
                entry, cache_dir=cache_dir, revision=revision, force_download=False,
                local_files_only=True, token=token,
            )
            return _checked(path)
        except FoundationModelDownloadError:
            if local_files_only:
                raise

    path = _hf_download(
        entry, cache_dir=cache_dir, revision=revision, force_download=force_download,
        local_files_only=local_files_only, token=token,
    )
    return _checked(path)


def _checked(path: str) -> str:
    """Guard the one assumption the rest of the framework makes about this path.

    ``Trainer._load_model`` branches on ``endswith('.pt')`` and, failing that,
    silently loads a different checkpoint, so a non-".pt" path here would
    fine-tune the wrong weights with only an INFO log.
    """
    if not str(path).endswith(".pt"):
        raise FoundationModelDownloadError(
            "resolved foundation model path %r does not end in '.pt'; the trainer "
            "would silently ignore it." % path
        )
    return path


def download_foundation_model(model_name: str, **kwargs) -> str:
    """
    Download a foundation model now and return its cached path.

    Identical to :func:`get_foundation_model_path`, named for intent: call it on a
    login node, or from rank 0, to warm the cache before a job that has no network.

    Examples
    --------
    >>> from iann.foundations import download_foundation_model
    >>> download_foundation_model("rpbe-all")
    """
    return get_foundation_model_path(model_name, **kwargs)


def is_cached(model_name: str) -> bool:
    """Whether `model_name` can be resolved without network access."""
    try:
        get_foundation_model_path(model_name, local_files_only=True)
        return True
    except (UnknownFoundationModelError, FoundationModelDownloadError):
        return False


def foundation_model_info(model_name: str) -> dict:
    """
    Metadata for a released foundation model.

    Returns the catalog record -- functional, training database, structure count,
    parameter count and reported MAEs -- plus whether it is currently cached.
    """
    entry = resolve_entry(model_name)
    if entry is None:
        raise _unknown_model_error(model_name)
    cached = is_cached(entry.name)
    return {
        "name": entry.name,
        "label": entry.label,
        "arch": entry.arch,
        "functional": entry.functional,
        "dataset": entry.dataset,
        "n_structures": entry.n_structures,
        "n_params": entry.n_params,
        "energy_mae": entry.energy_mae,
        "forces_mae": entry.forces_mae,
        "description": entry.description,
        "repo_id": HF_REPO_ID,
        "path": entry.path,
        "revision": HF_REVISION,
        "cached": cached,
        "local_path": get_foundation_model_path(entry.name, local_files_only=True)
        if cached
        else None,
    }


def list_foundation_models(groups=("painn", "arch")) -> List[str]:
    """
    Every released foundation model that can be requested by name.

    These are downloaded on first use. For the ones usable right now without
    network access, see :func:`list_available_models`.

    Examples
    --------
    >>> from iann.foundations import list_foundation_models
    >>> list_foundation_models(groups=("painn",))[:3]
    ['pbe-mptrj', 'pbe-salex', 'pbe-omat24']
    """
    return registry.all_names(groups)


def foundation_models_table(groups=("painn",)) -> str:
    """A plain-text table of the catalog, for the REPL and for documentation."""
    rows = registry.as_rows(groups)
    head = ("name", "functional", "database", "structures", "E MAE", "F MAE")
    widths = [max(len(head[0]), *(len(r["name"]) for r in rows)), 10, 34, 11, 7, 7]
    out = ["  ".join(h.ljust(w) for h, w in zip(head, widths)).rstrip(),
           "  ".join("-" * w for w in widths)]
    for r in rows:
        out.append("  ".join([
            r["name"].ljust(widths[0]),
            (r["functional"] or "").ljust(widths[1]),
            (r["dataset"] or "")[: widths[2]].ljust(widths[2]),
            ("" if r["n_structures"] is None else format(r["n_structures"], ",")).rjust(widths[3]),
            ("" if r["energy_mae"] is None else "%.4f" % r["energy_mae"]).rjust(widths[4]),
            ("" if r["forces_mae"] is None else "%.4f" % r["forces_mae"]).rjust(widths[5]),
        ]).rstrip())
    return "\n".join(out)


def load_foundation_model(model_name: str = "painn_oc.pt", device=None, **kwargs):
    """
    Load a foundation model checkpoint.

    Parameters
    ----------
    model_name : str, optional
        Model name, bundled filename, or path (see
        :func:`get_foundation_model_path`).
    device : str or torch.device, optional
        Device to map the checkpoint onto (default: 'cuda' if available).
    **kwargs
        Forwarded to :func:`get_foundation_model_path`.

    Returns
    -------
    dict
        The checkpoint.

    Examples
    --------
    >>> from iann.foundations.foundation_models import load_foundation_model
    >>> state_dict = load_foundation_model("rpbe-all")
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = get_foundation_model_path(model_name, **kwargs)
    # Checkpoints may have been downloaded, so refuse to execute arbitrary pickled
    # objects unless the restricted unpickler cannot read the file at all.
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        warnings.warn(
            "%s could not be loaded with weights_only=True and is being loaded with "
            "the unrestricted unpickler, which executes arbitrary code contained in "
            "the file. Only do this for checkpoints you trust." % model_path,
            RuntimeWarning,
            stacklevel=2,
        )
        state_dict = torch.load(model_path, map_location=device)
    return state_dict


def foundation_model(model_name: str = "painn_oc.pt", **kwargs) -> str:
    """
    Get the path to a foundation model, downloading it if necessary.

    Convenience wrapper around :func:`get_foundation_model_path` that resolves a
    model regardless of where the script is run from.

    Examples
    --------
    >>> from iann.foundations import foundation_model
    >>> from iann.calculators import MLCalculator
    >>>
    >>> # A released model, fetched on first use
    >>> calc = MLCalculator(foundation_model("rpbe-all"))
    >>>
    >>> # A checkpoint bundled with the package
    >>> calc = MLCalculator(foundation_model("painn_oc.pt"))
    >>>
    >>> # Or with default
    >>> calc = MLCalculator(foundation_model())
    """
    return get_foundation_model_path(model_name, **kwargs)


def list_available_models() -> List[str]:
    """
    Foundation models usable right now, without network access.

    That is: the checkpoints bundled inside this package, plus any released model
    already present in the local cache. For the full downloadable catalog, see
    :func:`list_foundation_models`.

    Examples
    --------
    >>> from iann.foundations import list_available_models
    >>> models = list_available_models()
    >>> print(models)
    """
    bundled = sorted(f for f in os.listdir(_HERE) if f.endswith('.pt'))
    cached = [n for n in registry.all_names() if is_cached(n)]
    return bundled + cached
