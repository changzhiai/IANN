"""One function per agent-facing operation, each returning a JSON-safe dict.

This is the only place the logic lives: :mod:`iann.agent.cli` adds argument
parsing and serialisation, :mod:`iann.agent.mcp_server` adds an MCP adapter, and
neither adds behaviour. That keeps the two front ends from drifting apart.

Conventions every function here follows:

* Return a plain ``dict`` of JSON-serialisable values (no numpy scalars, no
  tensors, no ``Path`` objects).
* Raise on failure, with a message that says what to do next. In particular
  ``export_lammps`` raises where the underlying converter returns ``None``.
* Import torch and the rest of the numerical stack **inside** the function.
  ``doctor`` must be able to report a broken environment, which it cannot do if
  importing this module already failed.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from typing import Any, Dict, List, Optional, Sequence

from .logparse import parse_log

# Architectures the LAMMPS converter accepts. Kept here rather than imported so
# that `iann models` works without torch; verified against the elif chain in
# iann/plugins/converter.py.
_EXPORTABLE = ("painn", "nequip", "allegro", "mace", "equiformerv2", "equiformerv3", "uma")

# Structural parameters the trainer does NOT persist in the checkpoint. Passing
# them at reconstruction time is mandatory when they differed from the defaults,
# otherwise load_state_dict fails with size mismatches. Keyed by architecture.
_UNPERSISTED_CONFIG = {
    "uma": ["num_distance_basis", "hidden_channels", "edge_channels", "mmax", "norm_type"],
    "equiformerv3": ["attn_grid_resolution_list", "ffn_grid_resolution_list", "mmax",
                     "norm_type", "attn_activation", "ffn_activation"],
    "equiformerv2": ["grid_resolution", "lmax_list", "mmax_list"],
    "allegro": ["num_scalar_features", "num_tensor_features"],
    "mace": [],
    "nequip": [],
    "painn": [],
    "fastpot": [],
    "demo": [],
}


def _jsonable(value: Any) -> Any:
    """Coerce numpy/torch scalars and arrays into built-in types."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):          # numpy array / torch tensor
        return _jsonable(value.tolist())
    if hasattr(value, "item"):            # numpy / torch scalar
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


# --------------------------------------------------------------- diagnostics

def doctor() -> Dict[str, Any]:
    """Report whether this interpreter can actually run IANN.

    Deliberately import-tolerant: every dependency is probed separately so the
    result distinguishes "torch missing" from "asap3 built against the wrong
    numpy", which is the failure this repo hits in a base conda env.
    """
    import warnings

    deps: Dict[str, Any] = {}
    for name in ("torch", "numpy", "ase", "e3nn", "asap3", "toml",
                 "huggingface_hub", "cpuinfo", "mcp"):
        entry: Dict[str, Any] = {"ok": False, "version": None, "error": None,
                                 "warnings": []}
        try:
            # Warnings are captured, not ignored: a half-broken install can
            # import "successfully" and still be unusable. torch against a
            # mismatched numpy is the case that matters here -- it imports and
            # then warns "Failed to initialize NumPy", which is otherwise
            # invisible to a caller checking only for exceptions.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                mod = __import__(name)
            entry["ok"] = True
            entry["version"] = getattr(mod, "__version__", None)
            entry["warnings"] = [str(w.message)[:200] for w in caught
                                 if "numpy" in str(w.message).lower()
                                 or "failed" in str(w.message).lower()]
        except Exception as exc:                       # noqa: BLE001 - reporting
            entry["error"] = f"{type(exc).__name__}: {exc}"
        deps[name] = entry

    cuda: Dict[str, Any] = {"available": False, "device_count": 0}
    if deps["torch"]["ok"]:
        import torch
        cuda = {"available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count())}

    # Can each architecture actually be constructed? This is the check that
    # catches a broken optional dependency behind one model only.
    architectures: Dict[str, Any] = {}
    try:
        from iann.models import MODEL_REGISTRY, get_model_class
        for arch in sorted(MODEL_REGISTRY):
            try:
                get_model_class(arch)
                architectures[arch] = {"importable": True, "error": None}
            except Exception as exc:                   # noqa: BLE001 - reporting
                architectures[arch] = {"importable": False,
                                       "error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:                           # noqa: BLE001 - reporting
        architectures = {"_error": f"{type(exc).__name__}: {exc}"}

    broken = [n for n, d in deps.items()
              if not d["ok"] and n not in ("mcp", "cpuinfo")]
    degraded = [n for n, d in deps.items() if d["ok"] and d["warnings"]]
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "dependencies": deps,
        "cuda": cuda,
        "architectures": architectures,
        "ok": not broken and not degraded,
        "broken": broken,
        "degraded": degraded,
        "hint": (None if not (broken or degraded) else
                 "This environment cannot run IANN reliably. A numpy ABI error from "
                 "asap3, or a 'Failed to initialize NumPy' warning from torch, means "
                 "the numerical stack is mismatched -- use the 'iann' conda env "
                 "rather than 'base'."),
    }


# ------------------------------------------------------------------ catalogue

def list_architectures() -> Dict[str, Any]:
    """The architectures the trainer accepts, and which can reach LAMMPS."""
    from iann.models import MODEL_REGISTRY
    names = sorted(MODEL_REGISTRY)
    return {
        "architectures": [
            {"name": n, "lammps_exportable": n in _EXPORTABLE,
             "unpersisted_config": _UNPERSISTED_CONFIG.get(n, [])}
            for n in names
        ],
        "count": len(names),
    }


def foundation_list(group: str = "painn") -> Dict[str, Any]:
    """Released foundation models. ``group`` is ``painn``, ``arch`` or ``all``."""
    from iann.foundations.registry import as_rows
    groups = ("painn", "arch") if group == "all" else (group,)
    rows = as_rows(groups=groups)
    return {"group": group, "models": _jsonable(rows), "count": len(rows)}


def foundation_info(name: str) -> Dict[str, Any]:
    """Everything known about one foundation model, including cache state."""
    from iann.foundations import foundation_model_info
    return _jsonable(foundation_model_info(name))


def foundation_fetch(name: str, *, local_files_only: bool = False) -> Dict[str, Any]:
    """Resolve a foundation model to a local path, downloading if needed."""
    from iann.foundations import foundation_model, is_cached
    was_cached = is_cached(name)
    path = foundation_model(name, local_files_only=local_files_only)
    return {"name": name, "path": path, "was_cached": bool(was_cached),
            "size_bytes": os.path.getsize(path) if os.path.isfile(path) else None}


# --------------------------------------------------------------- checkpoints

def inspect_checkpoint(model_path: str) -> Dict[str, Any]:
    """Report what a checkpoint records -- and what it is missing.

    The second half is the point: the trainer does not persist every structural
    parameter, so rebuilding some architectures needs values the file does not
    contain. ``missing_config`` names them instead of leaving the caller to
    discover it through a size-mismatch traceback.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No such checkpoint: {model_path}")

    import torch
    from iann.calculators.calculators import _infer_model_type

    state = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"{model_path} is not an IANN checkpoint "
                         "(expected a dict; a TorchScript export is not one)")

    declared = state.get("model_type")
    inferred = None
    try:
        inferred = _infer_model_type(state)
    except Exception:                                  # noqa: BLE001 - best effort
        pass
    arch = (declared or inferred or "").lower() or None

    meta = {k: _jsonable(v) for k, v in state.items()
            if k not in ("model", "optimizer", "scheduler") and not hasattr(v, "shape")}
    n_tensors = len(state["model"]) if isinstance(state.get("model"), dict) else 0

    return {
        "path": os.path.abspath(model_path),
        "size_bytes": os.path.getsize(model_path),
        "model_type_declared": declared,
        "model_type_inferred": inferred,
        "architecture": arch,
        "metadata": meta,
        "n_parameter_tensors": n_tensors,
        "missing_config": _UNPERSISTED_CONFIG.get(arch, []) if arch else [],
        "note": ("Values listed in missing_config are not stored in the checkpoint. "
                 "If they differed from the defaults during training, pass them when "
                 "rebuilding or exporting this model."),
    }


# --------------------------------------------------------------- computation

def predict(model_path: str, structure: str, *, index: int = 0,
            device: Optional[str] = None,
            ensemble: Optional[Sequence[str]] = None,
            config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Single-point energy and forces for one structure.

    With ``ensemble`` the variance across checkpoints is reported too, which is
    the signal to use for active learning.
    """
    if not os.path.isfile(structure):
        raise FileNotFoundError(f"No such structure file: {structure}")
    from ase.io import read

    atoms = read(structure, index=index)
    extra = dict(config or {})

    if ensemble:
        from iann.calculators import EnsembleCalculator
        paths = [model_path, *ensemble]
        for p in paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"No such checkpoint: {p}")
        atoms.calc = EnsembleCalculator(model_paths=paths, device=device, **extra)
    else:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"No such checkpoint: {model_path}")
        from iann.calculators import MLCalculator
        atoms.calc = MLCalculator(model_path=model_path, device=device, **extra)

    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    out: Dict[str, Any] = {
        "structure": os.path.abspath(structure),
        "index": index,
        "n_atoms": int(len(atoms)),
        "formula": atoms.get_chemical_formula(),
        "energy": energy,
        "energy_per_atom": energy / len(atoms),
        "max_force": float(abs(forces).max()),
        "forces": _jsonable(forces),
    }
    if ensemble:
        ens = atoms.calc.results.get("ensemble", {})
        out["ensemble"] = _jsonable(ens)
        out["n_models"] = 1 + len(ensemble)
    return out


def train(model: str, dataset: str, max_steps: int, *,
          output_dir: Optional[str] = None,
          config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Train for a bounded number of steps and return a manifest.

    ``max_steps`` is required and has no default. The trainer's own default is
    1,000,000 steps, so an agent that omitted a budget would start a run lasting
    days; requiring it here makes the cost an explicit decision. As a guide,
    PaiNN runs at roughly 1.6 s/step and NequIP at 4.1 s/step on CPU.

    Returns the final metrics from ``Trainer.eval_model()`` -- the library's
    ``train()`` returns ``None`` -- and writes the same manifest to
    ``<output_dir>/run.json``.
    """
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer; "
                         "it is required so that a run cannot be unbounded")
    if not os.path.isfile(dataset):
        raise FileNotFoundError(f"No such dataset: {dataset}")

    from iann.trainer import Trainer

    cfg: Dict[str, Any] = {"device": "cpu"}
    cfg.update(config or {})
    cfg["max_steps"] = max_steps
    if output_dir:
        cfg["output_dir"] = output_dir
    cfg.setdefault("output_dir", "output")
    cfg.setdefault("output_log", "output.log")
    cfg.setdefault("output_model", "model.pt")

    trainer = Trainer(model=model, config=cfg, distributed=False)
    trainer.train(dataset)

    try:
        metrics = _jsonable(trainer.eval_model())
    except Exception as exc:                           # noqa: BLE001 - reporting
        metrics = {"error": f"{type(exc).__name__}: {exc}"}

    out_dir = cfg["output_dir"]
    model_file = os.path.join(out_dir, cfg["output_model"])
    manifest = {
        "model": model,
        "dataset": os.path.abspath(dataset),
        "max_steps": max_steps,
        "config": _jsonable(cfg),
        "final_metrics": metrics,
        "output_dir": os.path.abspath(out_dir),
        "model_path": os.path.abspath(model_file) if os.path.isfile(model_file) else None,
        "log_path": os.path.abspath(os.path.join(out_dir, cfg["output_log"])),
        "progress": parse_log(os.path.join(out_dir, cfg["output_log"])),
    }
    try:
        with open(os.path.join(out_dir, "run.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)
        manifest["manifest_path"] = os.path.abspath(os.path.join(out_dir, "run.json"))
    except OSError as exc:
        manifest["manifest_path"] = None
        manifest["manifest_error"] = str(exc)
    return manifest


def training_status(output_dir: str) -> Dict[str, Any]:
    """Progress of a run, from its log and its checkpoint.

    Works while training is still going, which is the case an agent needs: the
    log gives the latest step and metrics, the checkpoint gives the best result
    so far.
    """
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"No such output directory: {output_dir}")

    progress = parse_log(os.path.join(output_dir, "output.log"))
    result: Dict[str, Any] = {"output_dir": os.path.abspath(output_dir), **progress}

    # The checkpoint is the only structured record of "best so far".
    ckpt = os.path.join(output_dir, "model.pt")
    result["checkpoint"] = None
    if os.path.isfile(ckpt):
        try:
            import torch
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            result["checkpoint"] = {
                "path": os.path.abspath(ckpt),
                "step": _jsonable(state.get("step")),
                "best_val_loss": _jsonable(state.get("best_val_loss")),
                "model_type": state.get("model_type"),
            }
        except Exception as exc:                       # noqa: BLE001 - reporting
            result["checkpoint"] = {"path": os.path.abspath(ckpt),
                                    "error": f"{type(exc).__name__}: {exc}"}

    manifest = os.path.join(output_dir, "run.json")
    result["manifest_path"] = os.path.abspath(manifest) if os.path.isfile(manifest) else None
    return result


def export_lammps(model_path: str, model_type: Optional[str] = None,
                  output_path: Optional[str] = None, *,
                  config: Optional[Dict[str, Any]] = None,
                  ensemble: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Export a checkpoint to TorchScript for ``pair_style iann``.

    Raises where the underlying converter merely returns ``None`` and prints --
    a missing input file would otherwise be reported to the caller as success.
    """
    paths = [model_path, *(ensemble or [])]
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"No such checkpoint: {p}")

    extra = dict(config or {})
    if ensemble:
        from iann.plugins.converter import convert_models_for_lammps
        if not model_type:
            raise ValueError("model_type is required for an ensemble export")
        result = convert_models_for_lammps(paths, model_type,
                                           output_path=output_path, **extra)
    else:
        from iann.plugins.converter import convert_model_for_lammps
        result = convert_model_for_lammps(model_path, model_type=model_type,
                                          output_path=output_path, **extra)

    if not result or not os.path.isfile(result):
        raise RuntimeError(
            f"Export produced no file for {model_path}. The converter reports "
            "failure by returning None; common causes are an unsupported "
            "architecture (FastPot cannot be exported) or structural config "
            "missing from the checkpoint -- run `iann inspect` to see which."
        )
    return {
        "model_path": os.path.abspath(model_path),
        "model_type": model_type,
        "output_path": os.path.abspath(result),
        "size_bytes": os.path.getsize(result),
        "n_models": len(paths),
    }
