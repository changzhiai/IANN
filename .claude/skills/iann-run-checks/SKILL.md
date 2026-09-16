---
name: iann-run-checks
description: Run IANN's test suite, docs build and LAMMPS export checks, and interpret their output correctly. Use when asked to verify a change, run the tests, build the documentation, or diagnose a failing check.
---

# Running the checks

All of these must be run **from the repository root** — paths are hardcoded.

## The test suite

```bash
python test/test_all.py              # ~17 min
python test/test_all.py --clean      # delete previous outputs first
```

Read the summary carefully, because two of its statuses do not mean what they
look like:

- **`VERIFIED` means the task hit the harness timeout** (120 s, or 300 s for
  EquiformerV2/Allegro/UMA), not that it converged. Every training script sets
  `max_steps: 30000000`, so training tasks always end this way. They are smoke
  tests: they prove the code runs, not that it learns.
- **Pass/fail is decided by scanning the last 1000 characters of child output**
  for `Error:`, `FAILED` or `Exception:`. Several child scripts legitimately
  print `FAILED` in their own summary tables, so a child that reports its
  failures gracefully can flip the parent to FAILED. Treat a FAILED as "read the
  log", not as a definite break. The log is `test/test_all.log`.

Not covered by the runner: `equiformerV3/train.py`, `fastpot/train.py`,
`demo/train.py`, every `predict.py`, and both docs scripts. Run those directly
if they are what you changed.

## The documentation

```bash
bash test/docs/install.sh            # one-time dependency install
bash test/docs/build.sh              # clean build + all checks
bash test/docs/build.sh --fast       # incremental, ~0.6 s, skips content checks
bash test/docs/build.sh --watch      # live reload on http://127.0.0.1:8000
bash test/docs/build.sh --open       # serve on 8001 and open a browser
```

Exit codes: 0 pass, 1 fail, 2 bad usage. Current baseline is **60 warnings, 0
errors**; the ceiling is `MAX_WARNINGS=86` because the count is Sphinx-version
dependent (86 on 7.3.7, 61 on 9.0.4).

It fails on: a non-zero sphinx exit, any docutils **ERROR** (these catch broken
references and malformed roles — a warning ceiling alone misses them), any
`autodoc: failed to import` (which otherwise renders empty API pages while
reporting success), more warnings than the ceiling, or a page missing content it
should carry.

Build output goes to `test/docs/output/`, never `docs/build/html` — that
directory is tracked, and building in place overwrites committed files.

Use `--open` rather than opening the HTML directly: Sphinx's search fetches over
HTTP and silently returns nothing under `file://`.

## LAMMPS exports

```bash
python test/lammps_plugin/export_models.py
```

Attempts all eight targets independently and prints a summary, then exits
non-zero if any failed. Each architecture is tried separately on purpose —
aborting on the first failure previously hid the fact that allegro, equiformerV3
and uma were never being exercised at all.

A `FAIL ... size mismatch` almost always means unpersisted structural config.
Run `iann inspect <ckpt>` to see what the checkpoint lacks.

## Environment first

If anything fails in a way that makes no sense, check the environment before
the code:

```bash
iann doctor
```

It exits non-zero on a broken environment and distinguishes a hard failure
(`asap3` cannot import) from a silent degradation (torch imports but warns that
NumPy failed to initialise). `base` shows both; use the `iann` conda env.

## A clean checkout has no checkpoints

`model.pt`, `*.log`, `test/*/*.traj` and anything matching `*output*` are
gitignored. The export, MD and JIT checks all read `test/<model>/output/model.pt`,
so they skip or fail until the corresponding training script has run. If a check
passes suspiciously fast, confirm the input actually exists.
