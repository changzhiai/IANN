---
name: iann-export-lammps
description: Export a trained IANN checkpoint to TorchScript for the LAMMPS pair_style iann, including the ensemble variant. Use when asked to deploy a model to LAMMPS, run production MD, or when a TorchScript export fails with size mismatches.
---

# Exporting to LAMMPS

```bash
iann export --model test/painn/output/model.pt --type painn \
            --out export_painn.pt --json
```

`--type` may be omitted and will be inferred from the checkpoint.

## The trap: checkpoints do not record every parameter

This is the failure you will actually hit. `num_distance_basis` and the
grid-resolution lists are **not persisted**, so the model is rebuilt at the
defaults and `load_state_dict` fails with `size mismatch`. Ask first:

```bash
iann inspect test/uma/output/model.pt --json    # see "missing_config"
```

Then pass the values used at training time in a TOML file via `--config`.
PaiNN, NequIP and MACE usually need nothing extra. EquiformerV3 and UMA usually do:

```toml
# UMA
num_layers = 2
num_channels = 32
lmax = 2
mmax = 2
hidden_channels = 32
edge_channels = 32
num_distance_basis = 128
cutoff = 5.5
norm_type = "rms_norm_sh"
norm_data = true
norm_per_atom = true
```

```toml
# EquiformerV3
num_layers = 2
num_channels = 16
lmax = 3
mmax = 2
attn_grid_resolution_list = [12, 6]
ffn_grid_resolution_list = [12, 12]
norm_type = "merge_layer_norm"
attn_activation = "sep-merge_gates2_swiglu"
ffn_activation = "sep-merge_gates2_swiglu"
use_envelope = true
norm_data = true
norm_per_atom = true
```

Working examples for all eight targets live in
`test/lammps_plugin/export_models.py`; running it prints a per-architecture
pass/fail summary and exits non-zero if any failed.

## What can and cannot be exported

Exportable: `painn`, `nequip`, `allegro`, `mace`, `equiformerv2` (also spelled
`equiformer2`), `equiformerv3`, `uma`.

- **FastPot cannot be exported.** `convert_model_for_lammps` raises
  `ValueError: Unknown model type: fastpot`.
- **UMA requires `num_experts == 0`** (the default). A model trained with MoLE
  experts is refused rather than exported without its routing, which would
  silently change the physics.

## Using it in LAMMPS

```
pair_style iann painn export_painn.pt 5.5
pair_coeff * *
```

The first argument is a label only — the C++ never compares it — but keep it
honest. For multi-GPU inference use `pair_style iann/multi_gpu`, which is worth
it above roughly 10,000 atoms; below ~5,000 atoms it is slower than serial
because of ghost-atom duplication. See `docs/source/performance.rst`.

Ensembles, for on-the-fly uncertainty:

```bash
iann export --model m1.pt --ensemble m2.pt m3.pt --type painn --out ens.pt
```

## Verification

Always check the export reproduces the eager model rather than assuming:

```bash
iann predict --model test/painn/output/model.pt --structure test/Pt_ads.traj --json
```

then load the TorchScript file and compare. `test/lammps_plugin/run_jit.py` does
exactly this. Expect agreement to ~1e-7 in energy.

**Seed the RNG before comparing EquiformerV3.** It picks a random third axis for
its rotation frame, so two identical eager calls already differ by ~4e-3 eV/Å in
forces; without a fixed seed you will misread that as an export bug.
