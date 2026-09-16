---
name: iann-hpc-submit
description: Write a SLURM or PBS submission script for distributed IANN training on an HPC cluster. Use when asked to run training on NERSC, S3DF, Carbon, or any multi-GPU or multi-node machine.
---

# Submitting distributed training

**Write the script; do not submit it.** Generate the file, show it, and leave
`sbatch`/`qsub` to the user — an allocation is theirs to spend.

## The training script never changes

Distributed training is configured from the scheduler environment, so `train.py`
only needs `distributed=True`. The trainer reads rank and world size from
`SLURM_PROCID`/`SLURM_NTASKS`, or from `OMPI_COMM_WORLD_RANK`/`_SIZE` (also
MPICH/Intel MPI's `PMI_*` and MVAPICH2's `MV2_*`).

```python
from iann.trainer import Trainer

trainer = Trainer(
    model="painn",
    config={"device": "cuda", "max_steps": 200000,
            "output_dir": "output", "output_log": "output.log",
            "output_model": "model.pt"},
    distributed=True,
)
trainer.train("dataset.traj")
```

## SLURM

```bash
#!/bin/bash
#SBATCH -N 2                   # nodes
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -A <account>
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4    # one rank per GPU
#SBATCH --cpus-per-task=1

export PYTHONPATH=/path/to/IANN:$PYTHONPATH
module load PrgEnv-nvidia; module load openmpi

export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
export NNODES=$SLURM_NNODES

srun -N $NNODES -n $((NNODES*GPUS_PER_NODE)) python train.py
```

On NERSC add the two vendor workarounds for multi-node runs:

```bash
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216
```

## PBS (e.g. Carbon)

PBS has no `srun`, so ranks are launched with `mpirun`:

```bash
#!/bin/bash
#PBS -l nodes=1:ppn=4:gpus=2
#PBS -l walltime=5:00:00
#PBS -N train
#PBS -A <account>
#PBS -o job.out
#PBS -e job.err

cd $PBS_O_WORKDIR
source ~/.bashrc
source /path/to/miniconda/etc/profile.d/conda.sh
conda activate base
module load openmpi

export PYTHONPATH=/path/to/IANN:$PYTHONPATH

GPUS_PER_NODE=2
export MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
export MASTER_PORT=12356

# mpirun does not carry the shell environment to remote nodes
FWD="-x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x MASTER_ADDR -x MASTER_PORT"

mpirun --map-by ppr:${GPUS_PER_NODE}:node -machinefile "$PBS_NODEFILE" $FWD python train.py
```

Three things differ from SLURM and all three matter:

1. **Rank comes from MPI**, not the scheduler — the trainer falls back to the
   `OMPI_COMM_WORLD_*` variables automatically.
2. **`--map-by ppr:N:node` replaces `-np`** — it places N ranks on each node in
   `$PBS_NODEFILE` without computing the total by hand.
3. **The environment must be forwarded with `-x`** — otherwise remote ranks start
   on the system Python and fail to import `iann`. This is the most common
   failure.

`MASTER_ADDR` is optional on PBS (the trainer reads the first host from
`$PBS_NODEFILE`), but if you do set it, it **must** be forwarded — a rank that
does not receive it picks its own master and the job hangs.

## Scaling expectations

Measured on the curated MPtrj database, 1→16 A100s: speedup 10.5–13.7×, parallel
efficiency 65.7–86.0 %, throughput 323–3,892 structures/s depending on
architecture. MACE, NequIP and PaiNN scale best. If scaling disappoints, check
the data loader before the network — neighbour-list construction is one of the
three dominant costs per step.

Full detail: `docs/source/parallelization.rst` and `docs/source/performance.rst`.

## Before submitting

Confirm the run is bounded (`max_steps` or `max_epochs` is set), the dataset path
exists on the cluster filesystem, and `output_dir` is writable from every node.
Then hand the script over and let the user submit.
