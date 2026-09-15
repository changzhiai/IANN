import os
import tempfile

from iann.foundations import (
    UnknownFoundationModelError,
    foundation_model,
    foundation_model_info,
    is_cached,
    list_available_models,
    list_foundation_models,
)
from iann.foundations.registry import FOUNDATION_MODELS, normalize_name
from iann.calculators import MLCalculator
from ase.build import fcc100


def test_catalog():
    """Offline checks on the catalog itself. These need no network."""
    print("\n--- Catalog ---")

    names = [e.name for e in FOUNDATION_MODELS.values()]
    paths = [e.path for e in FOUNDATION_MODELS.values()]
    assert len(names) == len(set(names)), "duplicate model name in the catalog"
    assert len(paths) == len(set(paths)), "duplicate repo path in the catalog"
    print(f"{len(names)} entries, names and paths unique")

    # Name normalisation: the manuscript spelling, the snake_case form and a
    # trailing .pt must all land on the same entry.
    assert normalize_name("PBE-MPtrj") == "pbe-mptrj"
    assert normalize_name("pbe_mptrj") == "pbe-mptrj"
    assert normalize_name("PBE-MPtrj.pt") == "pbe-mptrj"
    print("name normalisation ok")

    # An unknown name must fail fast, before any network or filesystem work.
    try:
        foundation_model("does-not-exist-at-all")
        raise AssertionError("expected UnknownFoundationModelError")
    except UnknownFoundationModelError as err:
        assert "does-not-exist-at-all" in str(err)
    print("unknown name raises UnknownFoundationModelError")

    # An explicit path to an existing file passes straight through.
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        handle.write(b"not-a-real-checkpoint")
        tmp = handle.name
    try:
        assert foundation_model(tmp) == os.path.abspath(tmp)
        print("explicit path passthrough ok")
    finally:
        os.unlink(tmp)

    # Metadata is present for every released model.
    for name in list_foundation_models(groups=("painn",)):
        info = foundation_model_info(name)
        assert info["functional"], f"{name} has no functional"
        assert info["energy_mae"] is not None, f"{name} has no energy MAE"
    print(f"metadata present for all {len(list_foundation_models(groups=('painn',)))} released models")


def test_foundation_models():
    """Load every locally available foundation model and evaluate one structure.

    `list_available_models()` is deliberately local-only -- bundled checkpoints
    plus anything already in the cache -- so this stays an offline smoke test
    rather than downloading the whole catalog.
    """
    models = list_available_models()
    print(f"\n--- Found {len(models)} locally available foundation models: {models} ---")

    atoms = fcc100("Pt", size=(4, 4, 3), a=5.5, vacuum=15.0)

    success_count = 0
    for model_name in models:
        print(f"\n--- Testing Foundation Model: {model_name} ---")
        try:
            calc = MLCalculator(
                model_path=foundation_model(model_name),
                compute_forces=True,
                device='cpu'
            )
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            print(f"SUCCESS: {model_name} Energy: {energy:.4f} eV")
            success_count += 1

        except Exception as e:
            print(f"FAILED: {model_name} - Error: {e}")

    print(f"\n--- Foundation Model Testing Summary: {success_count}/{len(models)} passed ---")
    if success_count < len(models):
        exit(1)


def test_download():
    """Resolve one released model by name, if it is cached or a network is up.

    Skipped rather than failed when offline, so the suite still runs on a compute
    node with no egress.
    """
    name = "rpbe-all"
    print(f"\n--- Resolving released model {name!r} ---")
    if not is_cached(name):
        print(f"SKIP: {name} is not cached and may need a download")
        return
    path = foundation_model(name)
    assert path.endswith(".pt"), "the trainer requires a .pt path"
    assert os.path.isfile(path)
    print(f"SUCCESS: {name} -> {path}")
    # A second call must be served from the cache, not re-downloaded.
    assert foundation_model(name) == path
    print("second call served from cache")


if __name__ == "__main__":
    test_catalog()
    test_download()
    test_foundation_models()
