---
name: iann-foundation-models
description: Find, download and use IANN's released foundation models, including fine-tuning them and working offline on compute nodes. Use when asked which pretrained model to start from, to fetch one, or to run without network access.
---

# Foundation models

Twelve pretrained PaiNN checkpoints, grouped by exchange–correlation functional
and trained on 41.9 million curated DFT structures. They are **priors** for
fine-tuning on a small in-house dataset, not finished potentials.

```bash
iann foundation list --json              # the twelve, with accuracies
iann foundation list --group all         # plus 7 architecture-comparison runs
iann foundation info rpbe-oc20 --json     # details + whether it is cached
iann foundation fetch rpbe-oc20           # resolve to a path, downloading if needed
```

## Choosing one

Match the **functional** first — absolute energies are not transferable across
functionals, which is why the models are grouped rather than pooled.

| Prefix | Functional | Use for |
|---|---|---|
| `pbe-*` | PBE | bulk materials; widest chemical coverage |
| `rpbe-*` | RPBE | **surfaces, adsorption, catalysis** |
| `r2scan-*` | r²SCAN | meta-GGA accuracy; smallest databases |

Then match the chemistry. RPBE has no actinides and only six lanthanides, but 17
million oxygen-containing and 12 million hydrogen-containing structures — far
more surface chemistry than its structure count suggests. f-block systems are
only covered at PBE and r²SCAN.

`*-all` merges every database at that functional: broader scope, but worse on
energies than the best of its constituents. Prefer a specialised checkpoint when
one matches your system.

Names are matched case-insensitively and accept the paper's spellings, so
`PBE-MPtrj`, `pbe_mptrj` and `pbe-mptrj` all resolve to the same entry. A typo
raises `UnknownFoundationModelError` with a did-you-mean list before any I/O.

## Fine-tuning

```toml
load_model = "/path/from/fetch/mp.pt"
reset_lr = true          # fresh LR schedule -- see below
learning_rate = 0.0001
```

`reset_lr` matters: without it the learning-rate schedule and step counter
resume from where pre-training stopped, so the run starts at a tiny learning
rate and barely moves. The released models are PaiNN, so use `--model painn`.

## Working offline

Compute nodes often have no network. Prefetch on the login node, then pin the
cache and forbid network access inside the job:

```bash
# login node
iann foundation fetch rpbe-oc20

# in the job script
export HF_HOME=/scratch/$USER/hf          # or IANN_FOUNDATION_CACHE
export HF_HUB_OFFLINE=1
```

`iann foundation fetch --local-files-only` fails loudly rather than reaching the
network, which is what you want inside a job. `iann foundation info` reports
`cached` and `local_path` so you can verify before submitting.

Set `HF_HOME` to scratch on HPC — the default `~/.cache/huggingface` lands on a
quota-limited home directory.

## Accuracy caveat

The reported MAEs are evaluated on the database each model was trained on, after
outlier removal, so they measure fit rather than extrapolation. The curated
databases were cleaned using a trained model's residuals, which makes them good
for training but **unsuitable for benchmarking** — the removed structures are
exactly the hard ones. Use the original published datasets for that.
