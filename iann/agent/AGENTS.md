# IANN — working notes for agents

IANN trains and deploys machine-learning interatomic potentials (MLIPs). Seven
equivariant graph-neural-network architectures sit behind one data object, one
trainer and one deployment route, so choosing a model is a one-word change in a
config dict.

This file records only what cannot be inferred from the code. Nothing in it is
specific to any one assistant — it is the repository knowledge any AI coding
agent needs, and any human is welcome to read it too.

It lives at `iann/agent/AGENTS.md`. `iann agent install` copies it to a
repository root under both the names agents look for: `AGENTS.md`, the
cross-tool convention, and `CLAUDE.md`, which is the only name Claude Code
reads. The two are identical; edit `iann/agent/AGENTS.md`, not an installed
copy.

## Environment

Use the `iann` conda environment, not `base`:

```bash
/opt/anaconda3/envs/iann/bin/python
```

`base` has a mismatched numerical stack: `asap3` fails with
`numpy.core.multiarray failed to import`, and torch imports but warns
`Failed to initialize NumPy`, which leaves it half-working in confusing ways.

Check before doing anything else — it exits non-zero on a bad environment:

```bash
iann doctor            # or: python -m iann.agent.cli doctor
```

## Run from the repository root

Scripts hardcode relative paths (`test/Pt_ads.traj`, `output_dir='test/painn/output'`).
Running them from anywhere else fails or writes to the wrong place.

## Bound every training run

`DEFAULT_CONFIG["max_steps"]` is **1,000,000**, and every script under `test/`
sets 30,000,000 — they only stop because the test harness kills them. Always
pass a step budget. `iann train` requires `--max-steps` for this reason.

Measured cost on CPU: PaiNN ≈1.6 s/step, NequIP ≈4.1 s/step.

## Checkpoints are not self-describing

The trainer does not persist every structural parameter. `num_distance_basis`
and the grid-resolution lists are absent, so rebuilding EquiformerV3 or UMA
needs the values used at training time, or `load_state_dict` fails with size
mismatches. Ask the checkpoint what it is missing:

```bash
iann inspect test/uma/output/model.pt
```

## Facts worth knowing

- **Architectures**: painn, nequip, allegro, mace, equiformerv2, equiformerv3, uma,
  plus fastpot and demo (internal). All except fastpot and demo export to LAMMPS.
- **EquiformerV3 is nondeterministic**: `init_edge_rot_mat` picks a random third
  axis, so forces vary by ~4e-3 eV/Å between identical calls. Seed the RNG before
  comparing anything.
- **UMA export requires `num_experts == 0`** (the default). A model trained with
  MoLE experts is refused rather than exported without its routing.
- **Docs**: build with `bash test/docs/build.sh`, never into `docs/build/html`
  (that directory is tracked). `--fast` for iteration, `--watch` for live reload.
- **`VERIFIED` in `test/test_all.py` means the run timed out**, not that it
  converged.
- **Gitignored**: `model.pt`, `*.log`, `test/*/*.traj`, anything matching `*output*`.
  A clean checkout therefore has no checkpoints — train one first, or fetch a
  foundation model.

## Common tasks

```bash
iann doctor                                          # is this env usable?
iann models                                          # architectures + export support
iann foundation list                                 # the 9 released models
iann inspect <ckpt>                                  # metadata + missing config
iann predict --model <ckpt> --structure <file>       # single-point E and F
iann train --model painn --dataset test/Pt_ads.traj --max-steps 20
iann status --output-dir output                      # progress of a running job
iann export --model <ckpt> --type painn --out m.pt   # TorchScript for LAMMPS
```

Add `--json` to any of them for parseable output. Exit codes: 0 success,
1 failure, 2 usage.
