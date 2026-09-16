---
name: iann-train
description: Train or fine-tune an IANN interatomic potential, choose an architecture, bound the run, and read back its metrics. Use when asked to train a model, compare architectures, or fine-tune a foundation model on new data.
---

# Training an IANN potential

## Always bound the run

The trainer's default is 1,000,000 steps and every script under `test/` asks for
30,000,000. Never start a run without a budget:

```bash
iann train --model painn --dataset test/Pt_ads.traj --max-steps 20 --out /tmp/run1 --json
```

`--max-steps` is required by this command precisely so an unbounded run cannot
be started by omission. Budget with the measured CPU cost: **PaiNN ≈1.6 s/step,
NequIP ≈4.1 s/step**; EquiformerV2 is slower still.

The result is a manifest — also written to `<out>/run.json` — with the config
used, the steps actually run, the final `eval_model()` metrics, and the paths of
the checkpoint and log. The library's own `Trainer.train()` returns `None`, so
this manifest is the only structured record of a run.

## Choosing an architecture

`iann models` lists them. From the paper's comparison on 1.58 M MPtrj structures:

| Architecture | E MAE (eV/atom) | F MAE (eV/Å) | Notes |
|---|---|---|---|
| PaiNN | 0.041 | 0.050 | fastest; 9.0 ms latency floor. The released foundation models use it |
| NequIP | 0.050 | 0.069 | |
| Allegro | 0.040 | 0.066 | strictly local; high marginal cost per atom |
| MACE | 0.054 | 0.069 | lowest marginal cost |
| EquiformerV2 | 0.040 | 0.059 | 115 ms latency floor; runs out of memory at 864 atoms on 32 GB |
| EquiformerV3 | 0.040 | 0.046 | |
| UMA | 0.029 | 0.040 | most accurate here |

Start with PaiNN unless there is a reason not to: it is the cheapest to train
and to run, and it is competitive on accuracy.

## Passing extra parameters

Anything beyond `num_channels`, `num_layers` and `cutoff` is architecture
specific and is forwarded verbatim to the model constructor. Put it in a TOML
file and pass `--config`:

```toml
num_channels = 32
num_layers = 2
batch_size = 4
log_interval = 1
learning_rate = 0.001
```

The full list of trainer keys and their defaults is `DEFAULT_CONFIG` in
`iann/trainer/trainer.py`, mirrored in `docs/source/training.rst`.

## Watching a run

```bash
iann status --output-dir /tmp/run1 --json
```

Reports the latest step and metrics from the log plus `best_val_loss` from the
checkpoint, and a `state` of `running`, `early_stopped`, `max_steps_reached` or
`nan_exit`. Safe to call while training is still going.

Note the log is opened in append mode, so one file can hold several sessions;
`status` reports the most recent.

## Fine-tuning a foundation model

Load the pretrained weights and start a **fresh** learning-rate schedule —
without `reset_lr` the schedule resumes where pre-training left off, which is
almost never what you want when adapting to new chemistry:

```bash
iann foundation fetch rpbe-all          # prints the cached path
```

```toml
load_model = "/path/from/fetch/mp.pt"
reset_lr = true
learning_rate = 0.0001
```

The released models are PaiNN, so fine-tuning them requires `--model painn`.

## Verification

After any run: `iann status` should report the expected step and a terminal
state, and `<out>/model.pt` should load — check with `iann inspect <out>/model.pt`.
