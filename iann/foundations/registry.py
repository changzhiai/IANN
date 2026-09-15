"""Static catalog of the IANN foundation models hosted on the HuggingFace Hub.

This module is deliberately dependency-free -- no torch, no huggingface_hub -- so
that it imports in microseconds and can be used to build documentation tables and
error messages without pulling in the rest of the framework.

Accuracy figures are those reported in the IANN paper (Table 3 for the released
PaiNN foundation models, Table 2 for the architecture comparison). They are
evaluated on the database each model was trained on, after outlier removal, so
they measure fit rather than extrapolation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

# The Hub repository holding the checkpoints.
HF_REPO_ID = "changzhiai/iann-foundation-models"
HF_REPO_TYPE = "model"

# Pinned to a commit rather than a branch: on a moving branch a later push to the
# Hub silently changes which weights a name resolves to, which would break
# reproducibility for any published result that cites a model by name. Update this
# deliberately, together with the accuracy figures below.
#
# This is the revision the catalog paths below were verified against. Set it to
# "main" only for local experimentation.
HF_REVISION = "bec4842d17e1478adc6ba8df0c920d7590c53501"


@dataclass(frozen=True)
class ModelEntry:
    """One downloadable checkpoint."""

    name: str                           # canonical key, e.g. "pbe-mptrj"
    label: str                          # display name, e.g. "PBE-MPtrj"
    path: str                           # repo-relative, e.g. "painn/pbe/mp/mp.pt"
    arch: str = "painn"
    functional: str = ""
    dataset: str = ""
    n_structures: Optional[int] = None
    n_params: Optional[int] = None
    energy_mae: Optional[float] = None  # eV/atom
    forces_mae: Optional[float] = None  # eV/Angstrom
    group: str = "painn"                # "painn" | "arch"
    description: str = ""


_E = ModelEntry
_PAINN = 600_577

_ENTRIES: List[ModelEntry] = [
    # -- Released PaiNN foundation models, grouped by functional (paper Table 3).
    #    All use 128 channels, 3 layers and a 5.5 A cutoff.
    _E("pbe-mptrj", "PBE-MPtrj", "painn/pbe/mp/mp.pt",
       functional="PBE", dataset="MPtrj", n_structures=1_579_058,
       n_params=_PAINN, energy_mae=0.0517, forces_mae=0.0498),
    _E("pbe-salex", "PBE-sAlex", "painn/pbe/alex/alex.pt",
       functional="PBE", dataset="sAlex", n_structures=10_422_893,
       n_params=_PAINN, energy_mae=0.0408, forces_mae=0.0513),
    _E("pbe-omat24", "PBE-OMat24", "painn/pbe/omat24/omat24.pt",
       functional="PBE", dataset="OMat24", n_structures=11_261_400,
       n_params=_PAINN, energy_mae=0.0261, forces_mae=0.1382),
    _E("pbe-matpes", "PBE-MatPES", "painn/pbe/matpes/matpes.pt",
       functional="PBE", dataset="MatPES", n_structures=434_083,
       n_params=_PAINN, energy_mae=0.0463, forces_mae=0.0930),
    _E("pbe-all", "PBE-all", "painn/pbe/all/pbe.pt",
       functional="PBE", dataset="MPtrj + sAlex + OMat24 + MatPES",
       n_structures=23_694_325, n_params=_PAINN,
       energy_mae=0.0545, forces_mae=0.1039,
       description="Merged PBE model, spanning the scope of all four sources."),

    _E("rpbe-oc20", "RPBE-OC20", "painn/rpbe/oc20/oc20.pt",
       functional="RPBE", dataset="OC20", n_structures=1_999_216,
       n_params=_PAINN, energy_mae=0.0263, forces_mae=0.0630),
    _E("rpbe-oc22", "RPBE-OC22", "painn/rpbe/oc22/oc22.pt",
       functional="RPBE", dataset="OC22", n_structures=8_198_695,
       n_params=_PAINN, energy_mae=0.0381, forces_mae=0.0528),
    _E("rpbe-oc25", "RPBE-OC25", "painn/rpbe/oc25/oc25.pt",
       functional="RPBE", dataset="OC25", n_structures=7_369_601,
       n_params=_PAINN, energy_mae=0.0116, forces_mae=0.0753),
    _E("rpbe-all", "RPBE-all", "painn/rpbe/all/rpbe.pt",
       functional="RPBE", dataset="OC20 + OC22 + OC25",
       n_structures=17_553_809, n_params=_PAINN,
       energy_mae=0.0454, forces_mae=0.0712,
       description="Merged RPBE model; the usual prior for adsorption energetics."),

    _E("r2scan-mptrj", "r2SCAN-MPtrj", "painn/r2scan/mptrj/mptrj.pt",
       functional="r2SCAN", dataset="MPtrj (r2SCAN subset)", n_structures=238_241,
       n_params=_PAINN, energy_mae=0.0206, forces_mae=0.0117),
    _E("r2scan-matpes", "r2SCAN-MatPES", "painn/r2scan/matpes/matpes.pt",
       functional="r2SCAN", dataset="MatPES (r2SCAN)", n_structures=387_285,
       n_params=_PAINN, energy_mae=0.0446, forces_mae=0.1127),
    _E("r2scan-all", "r2SCAN-all", "painn/r2scan/all/r2scan.pt",
       functional="r2SCAN", dataset="MPtrj + MatPES (r2SCAN)",
       n_structures=625_243, n_params=_PAINN,
       energy_mae=0.0678, forces_mae=0.0655,
       description="Merged r2SCAN model."),

    # -- Architecture comparison: seven architectures, one database (paper Table 2).
    _E("nequip-pbe-mptrj", "NequIP-PBE-MPtrj", "nequip/pbe/mp/mp.pt",
       arch="nequip", functional="PBE", dataset="MPtrj", group="arch",
       n_structures=1_579_091, n_params=1_379_513,
       energy_mae=0.050, forces_mae=0.069),
    _E("allegro-pbe-mptrj", "Allegro-PBE-MPtrj", "allegro/pbe/mp/mp.pt",
       arch="allegro", functional="PBE", dataset="MPtrj", group="arch",
       n_structures=1_579_091, n_params=567_624,
       energy_mae=0.040, forces_mae=0.066),
    _E("mace-pbe-mptrj", "MACE-PBE-MPtrj", "mace/pbe/mp/mp.pt",
       arch="mace", functional="PBE", dataset="MPtrj", group="arch",
       n_structures=1_579_091, n_params=1_429_376,
       energy_mae=0.054, forces_mae=0.069),
    _E("equiformerv2-pbe-mptrj", "EquiformerV2-PBE-MPtrj", "equiformerv2/pbe/mp/mp.pt",
       arch="equiformerv2", functional="PBE", dataset="MPtrj", group="arch",
       n_structures=1_579_091, n_params=773_089,
       energy_mae=0.040, forces_mae=0.059),
    _E("equiformerv3-pbe-mptrj", "EquiformerV3-PBE-MPtrj", "equiformerv3/pbe/mp/mp.pt",
       arch="equiformerv3", functional="PBE", dataset="MPtrj", group="arch",
       n_structures=1_579_091, n_params=943_249,
       energy_mae=0.040, forces_mae=0.046),
    _E("uma-pbe-mptrj", "UMA-PBE-MPtrj", "uma/pbe/mp/mp.pt",
       arch="uma", functional="PBE", dataset="MPtrj", group="arch",
       n_structures=1_579_091, n_params=1_173_698,
       energy_mae=0.029, forces_mae=0.040),
]

FOUNDATION_MODELS: Dict[str, ModelEntry] = {e.name: e for e in _ENTRIES}

# Checkpoints bundled inside the package. These predate the catalog above and are
# kept so that existing scripts and the published examples keep working.
BUNDLED_FILENAMES = (
    "painn_oc.pt",
    "painn_mptrj.pt",
    "painn_oc_124.pt",
    "painn_oc_132.pt",
)


def normalize_name(name: str) -> str:
    """Fold a user-supplied model name to its canonical form.

    Lowercases, trims, maps underscores to hyphens, and strips a trailing ".pt"
    only when the name is not path-like -- so "painn/pbe/mp/mp.pt" keeps its
    suffix and can be matched as a repo path.
    """
    out = str(name).strip().lower()
    if "/" not in out and "\\" not in out:
        out = out.replace("_", "-")
        if out.endswith(".pt"):
            out = out[: -len(".pt")]
    return out


def _build_aliases(entries: List[ModelEntry]) -> Dict[str, str]:
    """Map alternative spellings to canonical names.

    Derived from each entry rather than hand-listed, so an alias can never drift
    out of step with the repo path it points at. For "pbe-mptrj" this registers
    the full repo path, the repo directory, the display label, and the
    architecture-qualified form.
    """
    aliases: Dict[str, str] = {}
    for e in entries:
        for alt in (
            e.path,                                  # painn/pbe/mp/mp.pt
            e.path.rsplit("/", 1)[0],                # painn/pbe/mp
            e.label,                                 # PBE-MPtrj
            "%s-%s" % (e.arch, e.name),              # painn-pbe-mptrj
        ):
            key = normalize_name(alt)
            if key and key not in FOUNDATION_MODELS:
                aliases.setdefault(key, e.name)
    return aliases


ALIASES: Dict[str, str] = _build_aliases(_ENTRIES)


def resolve_entry(name: str) -> Optional[ModelEntry]:
    """Return the catalog entry for `name`, or None if it is not in the catalog."""
    key = normalize_name(name)
    if key in FOUNDATION_MODELS:
        return FOUNDATION_MODELS[key]
    if key in ALIASES:
        return FOUNDATION_MODELS[ALIASES[key]]
    return None


def all_names(groups=("painn", "arch")) -> List[str]:
    """Canonical names in catalog order, optionally restricted to some groups."""
    return [e.name for e in _ENTRIES if e.group in groups]


def as_rows(groups=("painn",)) -> List[dict]:
    """Catalog entries as plain dicts, for building documentation tables."""
    return [
        {
            "name": e.name,
            "label": e.label,
            "functional": e.functional,
            "dataset": e.dataset,
            "n_structures": e.n_structures,
            "n_params": e.n_params,
            "energy_mae": e.energy_mae,
            "forces_mae": e.forces_mae,
        }
        for e in _ENTRIES
        if e.group in groups
    ]
