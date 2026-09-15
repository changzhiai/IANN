#!/usr/bin/env bash
# Build the documentation and check it for regressions.
#
#   bash test/docs/build.sh            # build + check
#   bash test/docs/build.sh --open     # also open it in a browser
#   bash test/docs/build.sh --strict   # treat any warning as a failure
#   bash test/docs/build.sh --fast     # incremental rebuild, skip the checks
#   bash test/docs/build.sh --watch    # live reload on save (Ctrl-C to stop)
#
# Output goes to test/docs/output/, NOT to docs/build/html -- that directory is
# committed to the repository, so building in place would overwrite tracked files.
#
# Fails on:
#   * sphinx-build returning non-zero, or emitting SEVERE
#   * any "autodoc: failed to import" (a silent failure: the build still reports
#     success while the API pages render empty)
#   * more warnings than MAX_WARNINGS
#   * a page missing content it is supposed to carry
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/test/docs/output"
LOG="$OUT/sphinx.txt"
PY="${PYTHON:-python3}"

# All pre-existing and cosmetic: "Title underline too short" in installation.rst /
# lammps.rst / api.rst, duplicate object descriptions (api.rst and modules.rst
# document the same classes), and an ambiguous AtomsData cross-reference.
#
# The count is Sphinx-version dependent -- 86 on Sphinx 7.3.7, 61 on 9.0.4 -- so the
# ceiling is set by the noisier version rather than tightened to whichever one
# happens to be installed here. Override per-run with MAX_WARNINGS=61 to hold a
# newer Sphinx to its own baseline.
MAX_WARNINGS="${MAX_WARNINGS:-86}"

DO_OPEN=0
STRICT=0
FAST=0
WATCH=0
for arg in "$@"; do
  case "$arg" in
    --open)   DO_OPEN=1 ;;
    --strict) STRICT=1 ;;
    --fast)   FAST=1 ;;
    --watch)  WATCH=1 ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

if ! "$PY" -c 'import sphinx, sphinx_rtd_theme' 2>/dev/null; then
  echo "sphinx and/or sphinx_rtd_theme are not installed in this environment."
  echo "run:  bash test/docs/install.sh"
  exit 1
fi

# --watch: hand over to sphinx-autobuild, which rebuilds on save and reloads the
# browser. Never returns; Ctrl-C to stop.
if [ "$WATCH" -eq 1 ]; then
  if ! "$PY" -c 'import sphinx_autobuild' 2>/dev/null; then
    echo "sphinx-autobuild is not installed. run:"
    echo "    $PY -m pip install sphinx-autobuild"
    exit 1
  fi
  mkdir -p "$OUT"
  echo ">>> watching $ROOT/docs/source -- open http://127.0.0.1:8000  (Ctrl-C to stop)"
  exec "$PY" -m sphinx_autobuild "$ROOT/docs/source" "$OUT" --port 8000
fi

# A clean build by default, so a stale artefact can never mask a problem.
# --fast keeps the cache, which turns a ~10 s rebuild into ~3 s while editing.
if [ "$FAST" -eq 0 ]; then
  rm -rf "$OUT"
fi
mkdir -p "$OUT"

echo ">>> building $ROOT/docs/source -> $OUT"
"$PY" -m sphinx -b html "$ROOT/docs/source" "$OUT" >"$LOG" 2>&1
BUILD_RC=$?

WARNINGS=$(grep -c 'WARNING' "$LOG" || true)
IMPORT_FAILS=$(grep -c 'autodoc: failed to import' "$LOG" || true)
SEVERE=$(grep -c 'SEVERE' "$LOG" || true)

echo
echo "    exit code            : $BUILD_RC"
echo "    warnings             : $WARNINGS  (max $MAX_WARNINGS)"
echo "    autodoc import fails : $IMPORT_FAILS  (must be 0)"
echo "    severe               : $SEVERE  (must be 0)"
echo "    log                  : $LOG"

FAIL=0
[ "$BUILD_RC" -ne 0 ]                 && { echo "FAIL: sphinx-build exited $BUILD_RC"; FAIL=1; }
[ "$SEVERE" -ne 0 ]                   && { echo "FAIL: severe messages present"; FAIL=1; }
[ "$IMPORT_FAILS" -ne 0 ]             && { echo "FAIL: autodoc could not import modules -- API pages will be empty"; FAIL=1; }
[ "$WARNINGS" -gt "$MAX_WARNINGS" ]   && { echo "FAIL: warning count rose above $MAX_WARNINGS"; FAIL=1; }
if [ "$STRICT" -eq 1 ] && [ "$WARNINGS" -ne 0 ]; then
  echo "FAIL: --strict and $WARNINGS warnings"; FAIL=1
fi

if [ "$IMPORT_FAILS" -ne 0 ] || [ "$SEVERE" -ne 0 ]; then
  echo
  echo "--- first few problems ---"
  grep -E 'autodoc: failed to import|SEVERE' "$LOG" | head -8
fi

# --fast skips the content checks: they are for catching regressions, not for the
# edit-and-look loop.
if [ "$FAST" -eq 1 ]; then
  echo
  if [ "$FAIL" -eq 0 ]; then echo "DOCS BUILD (fast): PASS"; else echo "DOCS BUILD (fast): FAIL"; fi
  [ "$DO_OPEN" -eq 1 ] && { command -v open >/dev/null && open "$OUT/index.html"; }
  exit "$FAIL"
fi

# Content checks: catch a page that builds but says the wrong thing, which no
# warning count would reveal.
echo
echo ">>> content checks"
"$PY" - "$OUT" <<'PY'
import html, pathlib, re, sys

out = pathlib.Path(sys.argv[1])

def text(page):
    """Visible text plus the raw markup, so a needle may be either prose or an
    attribute value such as an <img src=...> filename."""
    p = out / page
    if not p.is_file():
        return None
    raw = p.read_text()
    return html.unescape(re.sub(r"<[^>]+>", " ", raw)) + "\n" + raw

# page -> (label, [strings that must appear])
CHECKS = {
    "engine_models.html": ("all architectures documented",
        ["PaiNN", "NequIP", "Allegro", "MACE", "EquiformerV2", "EquiformerV3", "UMA", "FastPot"]),
    "training.html": ("model= values listed",
        ["painn", "nequip", "allegro", "mace", "equiformerv2", "equiformerv3", "uma", "fastpot"]),
    "foundation_models.html": ("released models listed",
        ["pbe-mptrj", "pbe-salex", "pbe-omat24", "pbe-matpes", "pbe-all",
         "rpbe-oc20", "rpbe-oc22", "rpbe-oc25", "rpbe-all",
         "r2scan-mptrj", "r2scan-matpes", "r2scan-all",
         "iann-foundation-models",
         "fig3_dft_db.png", "DFT databases", "dft-databases"]),
    "api.html": ("model classes in the API reference",
        ["PaiNN", "NequIP", "Allegro", "MACE", "EquiformerV2", "EquiformerV3", "UMA"]),
    "index.html": ("front page lists the architectures",
        ["PaiNN", "NequIP", "Allegro", "MACE", "EquiformerV2", "EquiformerV3", "UMA"]),
    "about.html": ("overview page with both paper figures",
        ["fig1_framework.png", "fig2_mechanism.png",
         "Framework structure", "Underlying mechanism", "10.5281/zenodo.17809949"]),
    "performance.html": ("cost page with both benchmark figures",
        ["fig6_models_cost.png", "fig7_lammps_scaling.png",
         "latency floor", "iann/multi_gpu", "Parallel efficiency"]),
}

# Strings that must NOT appear anywhere: stale repository names.
FORBIDDEN = ["dft-foundation-models", "dft_traj"]

bad = 0
for page, (label, needles) in CHECKS.items():
    t = text(page)
    if t is None:
        print(f"   FAIL  {page}: not built")
        bad += 1
        continue
    missing = [n for n in needles if n not in t]
    if missing:
        print(f"   FAIL  {page}: {label} -- missing {missing}")
        bad += 1
    else:
        print(f"   ok    {page}: {label} ({len(needles)} checked)")

for page in out.glob("*.html"):
    t = html.unescape(re.sub(r"<[^>]+>", " ", page.read_text()))
    raw = page.read_text()
    for f in FORBIDDEN:
        if f in t or f in raw:
            print(f"   FAIL  {page.name}: stale reference {f!r}")
            bad += 1

raise SystemExit(1 if bad else 0)
PY
CONTENT_RC=$?
[ "$CONTENT_RC" -ne 0 ] && FAIL=1

echo
if [ "$FAIL" -eq 0 ]; then
  echo "DOCS BUILD: PASS"
  [ "$DO_OPEN" -eq 1 ] && { command -v open >/dev/null && open "$OUT/index.html"; }
  exit 0
else
  echo "DOCS BUILD: FAIL  (see $LOG)"
  exit 1
fi
