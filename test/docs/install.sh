#!/usr/bin/env bash
# Install what is needed to build the documentation, into the active environment.
#
#   bash test/docs/install.sh
#
# Sphinx and the theme come from docs/requirements.txt. The remaining packages are
# needed because autodoc imports `iann` for real: conf.py mocks the heavy numerical
# stack (torch, e3nn, asap3, torch_geometric, opt_einsum_fx, cuequivariance) but not
# the rest of the import chain. Without them the build still "succeeds" while every
# model class fails to import and the API page renders empty.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-python3}"

echo "environment : $("$PY" -c 'import sys; print(sys.prefix)')"
echo "interpreter : $("$PY" -c 'import sys; print(sys.executable)')"
echo

echo ">>> installing docs requirements"
"$PY" -m pip install --quiet -r "$ROOT/docs/requirements.txt"

echo ">>> installing packages autodoc needs to import iann"
"$PY" -m pip install --quiet ase scipy numpy py-cpuinfo toml huggingface_hub

echo
echo ">>> checking"
"$PY" - <<'PY'
import importlib.util as u

REQUIRED = ["sphinx", "sphinx_rtd_theme"]
# Imported by the iann package itself, so autodoc needs them present.
FOR_AUTODOC = ["ase", "scipy", "numpy", "cpuinfo", "toml", "huggingface_hub"]
# Mocked by docs/source/conf.py, so they are NOT required to build the docs.
MOCKED = ["torch", "e3nn", "asap3", "torch_geometric", "opt_einsum_fx"]

missing = []
for group, mods in (("required", REQUIRED), ("autodoc", FOR_AUTODOC)):
    for m in mods:
        ok = u.find_spec(m) is not None
        print(f"   {'ok  ' if ok else 'MISS'}  {group:8s} {m}")
        if not ok:
            missing.append(m)
for m in MOCKED:
    state = "present" if u.find_spec(m) else "absent (fine, mocked)"
    print(f"   --    mocked   {m}  [{state}]")

if missing:
    raise SystemExit(f"\nstill missing: {', '.join(missing)}")
print("\nready: run  bash test/docs/build.sh")
PY
