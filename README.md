# IANN (InterAtomic Neural Network Framework)

[![Docs](https://img.shields.io/badge/Docs-available-blue)](https://iann.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
[![Python](https://img.shields.io/badge/Python-3.7%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/) 
[![C++](https://img.shields.io/badge/C++-11-00599C?logo=cplusplus&logoColor=white)](https://isocpp.org/) 


- [1. Introductions](#1-introductions)
  - [Documentation](#documentation)
- [2. Installation](#2-installation)
  - [Prerequisites](#prerequisites)
  - [Installing IANN](#installing-iann)
  - [GPU Support](#gpu-support)
- [3. Quickstart: Examples](#3-quickstart-examples)
- [4. Training](#4-training)
  - [Preparing your dataset](#preparing-your-dataset)
  - [Training](#training)
  - [Monitoring Training Progress](#monitoring-training-progress)
- [5. Predicting](#5-predicting)
  - [Making Predictions with ASE calculator](#making-predictions-with-ase-calculator)
- [6. Foundation Models](#6-foundation-models)
  - [Using Pre-trained Foundation Models](#using-pre-trained-foundation-models)
  - [Fine-tuning Foundation Models](#fine-tuning-foundation-models)
- [7. Parallelization](#7-parallelization)
  - [Multi-GPU Training](#multi-gpu-training)
  - [Multi-CPU Training](#multi-cpu-training)
  - [Example on NERSC](#example-on-nersc)
  - [Performance Considerations](#performance-considerations)
- [8. LAMMPS Interface](#8-lammps-interface)
  - [Use an IANN model with LAMMPS](#use-an-iann-model-with-lammps)
    - [1. Convert a trained model to the torchscript format](#1-convert-a-trained-model-to-the-torchscript-format)
    - [2. Use the exported model in LAMMPS](#2-use-the-exported-model-in-lammps)
  - [Use an ensemble IANN model with LAMMPS](#use-an-ensemble-iann-model-with-lammps)
    - [1. Convert trained models to a ensemble model](#1-convert-trained-models-to-a-ensemble-model)
    - [2. Use the exported ensemble model in LAMMPS](#2-use-the-exported-ensemble-model-in-lammps)
- [9. Modules](#9-modules)
  - [iann.data](#ianndata)
  - [iann.models](#iannmodels)
  - [iann.calculators](#ianncalculators)
  - [iann.plugins](#iannplugins)
  - [C++ LAMMPS Plugins](#c-lammps-plugins)
- [Troubleshooting](#troubleshooting)
- [Issues](#issues)
- [Maintainer](#maintainer)
- [References](#references)


## 1. Introductions

IANN (InterAtomic Neural Network) is an equivariant interatomic neural network potential framework package for materials science and computational chemistry. It implements state-of-the-art graph neural network models for periodic and non-periodic systems, including [FastPot](https://github.com/changzhiai/IANN), [PaiNN](https://arxiv.org/abs/2102.03150), [Nequip](https://doi.org/10.1038/s41467-022-29939-5), [MACE](https://arxiv.org/abs/2206.07697), and [EquiformerV2](https://arxiv.org/abs/2306.12059), focusing on predicting energies and forces with high accuracy. 

Key features:
- Easy to use and to switch models
- Multiple equivariant interatomic neural network models implementation
- High-accuracy energy and force predictions
- Distributed training on multiple GPUs and multiple server nodes
- Integration with ASE and LAMMPS for molecular dynamics simulations
- Customizable model architectures

### Documentation
A documentation is available at: https://iann.readthedocs.io

## 2. Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.9+

### Installing IANN

```bash
# Clone the repository
git clone https://github.com/changzhiai/IANN.git
cd IANN

# Install with pip
pip install -e .
```

### GPU Support
For GPU acceleration, make sure you have CUDA installed and PyTorch with CUDA support:

```bash
# Check if PyTorch is using CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## 3. Quickstart: Examples

The quickest way to get started with IANN is to run the example script:

```bash
# Run the quickstart example
python examples/quickstart.py
```

This script demonstrates:
- Loading a dataset
- Creating and training a model
- Using the model for predictions

Check out the `examples/` directory for more sample scripts and tutorials.

## 4. Training

### Preparing your dataset

IANN works with ASE database (.db) or trajectory (.traj) files. Ensure your data contains atomic structures with energy and force labels.

### Training
Create `train.py`

```python
from iann.trainer import Trainer

# Define the Trainer
trainer = Trainer(
    model="painn",
    config={"device": "cpu", 
            'output_dir': 'output',
            'output_log': 'output.log',
            'output_model': 'model.pt',
            },
    distributed=False
    )

# Run the training
trainer.train("dataset.traj")
```

Available models for `model`:
```
- fastpot
- painn
- nequip
- mace
- equiformerV2
```

Default configurations for `config`:
```python
config = {
    # parameters for model
    "num_channels": 128, # number of channels in the model
    "num_layers": 3, # number of layers in the model
    "cutoff": 5.5, # cutoff radius
    # parameters for trainer
    "device": None,      # override device, e.g. 'cpu' or 'cuda'
    "val_ratio": 0.1, # validation ratio
    "batch_size": 12, # batch size
    "learning_rate": 0.0001, # initial learning rate
    "forces_weight": 0.9, # weight for forces
    "load_model": False, # load model from checkpoint
    "max_steps": 1000000, # maximum number of steps
    "max_epochs": None,  # None if setup max_steps, otherwise max_epochs
    "optimizer_type": "adam", # optimizer type: "adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "adamw"
    "max_grad_norm": None,    # gradient clipping norm
    "log_interval": 2000, # log interval
    "stop_patience": 200, # patience for early stopping
    "scheduler_type": "LambdaLR", # scheduler type: "ReduceLROnPlateau", "LambdaLR", "CosineAnnealingLR", "CosineAnnealingWarmRestarts", "StepLR", "MultiStepLR", "ExponentialLR"
    # parameters for data
    "random_seed": 666, # random seed for reproducibility
    "save_split": False, # save split file name
    "load_split": False, # load split file name
    "norm_data": False, # normalize data
    "norm_per_atom": False, # normalize data per atom
    # parameters for DDP (Parallelization)
    "dist_timeout": 600,  # timeout (seconds) for distributed operations
    "master_port": 12356, # port for distributed operations
    # parameters for output
    "output_dir": "output", # output directory
    "output_log": "output.log", # log file
    "output_model": "model.pt", # model file
    "log_input": False, # log your costomized input config
    "debug": False, # debug mode
}
```

> [!NOTE]
> There are more parameters for each model, please refer to the documentation or source code for details.

### Monitoring Training Progress

Training logs will be saved in the specified output directory. You can monitor:
- Energy and force prediction errors
- Training and validation losses
- Model checkpoints

## 5. Predicting


### Making Predictions with ASE calculator

```python
from iann.calculators import MLCalculator
from ase.io import read

# Create calculator with model path
calc = MLCalculator("model.pt")

# Read structures
images = read("test_structures.traj", ":")

# Make predictions
for atoms in images:
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    print(f"Energy: {energy} eV")
    print(f"Forces: {forces} eV/Å")
```
>[!TIP]  
> `EnsembleCalculator` and `AtomicEnsembleCalculator` are available to get uncertainty for each structure and each atom, seperately.

## 6. Foundation Models

IANN provides pre-trained foundation models (painn) that you can use out-of-the-box or fine-tune for your specific tasks.

### Using Pre-trained Foundation Models

To use a foundation model for predictions:

```python
from iann.foundations import foundation_model
from iann.calculators import MLCalculator
from ase.build import fcc100

calc = MLCalculator(
  model_path=foundation_model("painn_oc.pt"), # foundation model trained on OC20+OC22
  compute_forces=True,
  device='cpu') # use 'cuda' for GPU

atoms = fcc100("Pt", size=(4,4,3), a=5.5, vacuum=15.0)
atoms.calc = calc
nnp_energy = atoms.get_potential_energy()
nnp_forces = atoms.get_forces()
print(f"NNP Energy: {nnp_energy:.4f} eV")
print(f"NNP Forces: {nnp_forces}")
```

### Fine-tuning Foundation Models

You can fine-tune a foundation model on your own data:

```python
from iann.trainer import Trainer
from iann.foundations import foundation_model

trainer = Trainer(model="painn", 
    config={"num_channels": 128, # number of channels in the model
        "num_layers": 3, # number of layers in the model
        "cutoff": 5.5, # cutoff radius
        "batch_size": 16, # batch size
        "learning_rate": 0.0001, # initial learning rate
        "forces_weight": 0.9, # weight for forces
        "load_model": foundation_model("painn_oc.pt"), # load the foundation model
        "max_steps": 10000000, # maximum number of steps
        "random_seed": 888, # random seed for reproducibility
        "val_ratio": 0.003, # validation ratio
        "stop_patience": 500, # patience for early stopping
        'device': 'cuda',
        'output_dir': 'output',
        'output_log': 'output.log',
        'output_model': 'model.pt'},
    distributed=False)
trainer.train("dataset.traj")
```

## 7. Parallelization

IANN supports distributed training using PyTorch's Distributed Data Parallel (DDP).

### Multi-GPU Training
Submit to multiple GPUs (in SLURM Workload Manager)

```bash
# Run on multiple GPUs and multiple nodes
#!/bin/bash
#SBATCH -N 2                   # Number of nodes
#SBATCH -C gpu                 # Use GPU nodes
#SBATCH -q debug               # Use regular/debug queue
#SBATCH -t 00:30:00            # Time limit
#SBATCH -A m2997               # Your account
#SBATCH --gpus-per-node=4      # GPUs per node
#SBATCH --ntasks-per-node=4    # Number of tasks per node
#SBATCH --cpus-per-task=1      # Number of CPUs per task

module load your_modules

export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
export NNODES=$SLURM_NNODES

srun -N $NNODES -n $((NNODES*GPUS_PER_NODE)) python train.py
```
### Multi-CPU Training
Submit to multiple CPUs (in SLURM Workload Manager)

```bash
# Run on multiple CPUs and multiple nodes
#!/bin/bash
#SBATCH -N 2                   # Number of nodes
#SBATCH -C cpu                 # Use CPU nodes
#SBATCH -q debug               # Use regular/debug queue
#SBATCH -t 00:30:00            # Time limit
#SBATCH -A m2997               # Your account
#SBATCH --ntasks-per-node=1    # Number of tasks per node
#SBATCH --cpus-per-task=128    # Number of CPUs per task

module load your_modules

export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
export NNODES=$SLURM_NNODES

srun -N $NNODES -n $((NNODES*GPUS_PER_NODE)) python train.py
```

### Example on NERSC
```bash
#!/bin/bash
#SBATCH -N 2                   # Number of nodes
#SBATCH -C gpu                 # Use GPU nodes
#SBATCH -q debug               # Use regular/debug queue
#SBATCH -t 00:20:00            # Time limit
#SBATCH -A m2997               # Your account
#SBATCH --gpus-per-node=4       # GPUs per node
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1

export PYTHONPATH=/pscratch/sd/c/changzhi/softwares/IANN_v2/IANN/:$PYTHONPATH
module purge
module load PrgEnv-nvidia; module load openmpi;

export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
export NNODES=$SLURM_NNODES
export FI_CXI_RDZV_GET_MIN=0 # vender bugs fixed on nersc for multiple nodes
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216 # vender bugs fixed on nersc

srun -N $NNODES -n $((NNODES*GPUS_PER_NODE)) \
     python train.py

```

> [!NOTE]
> the parallelization parameters are automatically obtained from the SLURM environment variables.

### Performance Considerations

- Use the largest batch size that fits in your GPU memory
- Enable mixed precision training for faster performance
- Monitor GPU utilization to ensure efficient resource use

## 8. LAMMPS Interface

IANN models can be used as interatomic potentials in LAMMPS molecular dynamics simulations (Support GPU).

> [!WARNING]
> You have to install IANN plugins for LAMMPS first, if you want to use IANN models with LAMMPS. Please see the documentation in [LAMMPS interface](https://iann.readthedocs.io/en/latest/lammps.html) section.

### Use an IANN model with LAMMPS


#### 1. Convert a trained model to the torchscript format

First, you need to have a trained model with torch format, which can be obtained by running the training script. Then convert the model to the torchscript format as follows:

```python
from iann.plugins.converter import convert_model_for_lammps

convert_model_for_lammps(model_path='best_model.pt', 
                         model_type='painn', 
                         output_path='output_model.pt')
```

#### 2. Use the exported model in LAMMPS

To run the LAMMPS simulation with IANN, you can use the following script:

```bash
# LAMMPS input script example

# Define the units and the atom style
units metal
atom_style atomic

# Define the boundary conditions
boundary p p p

# Read the initial structure
read_data initial.data

# Define the IANN pair style
pair_style iann painn model_lmp.pt 5.5
pair_coeff * *

# Define the mass of the atoms
mass 1 1.0079999997406976 # H
mass 2 195.08399994981576 # Pt

# Define the neighbor list
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes

# Thermodynamic settings
thermo 10

# Initial minimization to relax the system before dynamics
minimize 1.0e-4 1.0e-6 100 1000

# Define the timestep and the thermostat
timestep 0.001
fix 1 all nvt temp 300.0 300.0 0.1

# Define the dump frequency and the dump file
dump 1 all custom 10 dump.xyz id type x y z

# Run the simulation
run 5000
```

> [!NOTE]
> Multiple GPUs prediction (inference) are supported by using the `pair_style iann/multi_gpu` command. It will automatically detect the number of GPUs per node and use them to run the model.

### Use an ensemble IANN model with LAMMPS

#### 1. Convert trained models to a ensemble model

First, you need to have several trained models with torch format, which can be obtained by running several training scripts. Then convert the models to the torchscript format as follows:

```Python
from iann.plugins.converter import convert_models_for_lammps

# Give a list of models
model_paths = ["model_1.pt", "model_2.pt"]

# Convert the models to a torchscript model
output_path = convert_models_for_lammps(
    model_paths=model_paths,
    model_type="painn", # if not specified, the model type will be inferred from the model file
    output_path="model_ensemble_lmp.pt"
)
```
#### 2. Use the exported ensemble model in LAMMPS

To run the ensemble LAMMPS simulation with IANN, you can use the following script:

```bash
# LAMMPS input script example
   
# Define the units and the atom style
units metal
atom_style atomic

# Define the boundary conditions
boundary p p p

# Read the initial structure
read_data initial.data

# Define the IANN pair style
pair_style iann painn model_ensemble_lmp.pt 5.5
pair_coeff * *

# Define the mass of the atoms
mass 1 1.0079999997406976 # H
mass 2 195.08399994981576 # Pt

# Define the neighbor list
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes

# Compute the variance mode of the energy and force of the ensemble model
compute variance all iann/variance

# Define the thermodynamic style
thermo_style custom step pe ke etotal temp press c_variance[1] c_variance[2] c_variance[3] c_variance[4]

# Define the thermodynamic modify
thermo_modify colname c_variance[1] energy_var
thermo_modify colname c_variance[2] force_var
thermo_modify colname c_variance[3] max_energy_var
thermo_modify colname c_variance[4] max_force_var
thermo_modify flush yes

# Thermodynamic settings
thermo 100

# Initial minimization to relax the system before dynamics
minimize 1.0e-4 1.0e-6 100 1000

# Define the timestep and the thermostat
timestep 0.001
fix 1 all nvt temp 300.0 300.0 0.1

# Define the dump frequency and the dump file
dump 1 all custom 10 dump.xyz id type x y z

# Run the simulation
run 5000
```


## 9. Modules

IANN is organized into several key modules:

### iann.data
Data handling utilities:
- `AtomsData`: Data object for each atoms
- `AseDataset`: Dataset class for handling atomic structures

  
### iann.models
Contains neural network model implementations:
- `FastPot`: FastPot model implementation for energy and force prediction
- `PaiNN`: PaiNN model implementation for energy and force prediction
- `Nequip`: Nequip model implementation for energy and force prediction
- `MACE`: MACE model implementation for energy and force prediction
- `EquiformerV2`: EquiformerV2 model implementation for energy and force prediction


### iann.calculators
ASE calculators implementations:
- `MLCalculator`: ASE calculator interface for models
- `EnsembleCalculator`: ASE ensemble calculator interface for models
- `AtomicEnsembleCalculator`: ASE atomic ensemble calculator interface for models


### iann.plugins
Tools for converting models and LAMMPS integration:
- `converter`: Model conversion utilities for LAMMPS integration
- `EnsembleLAMMPSModelWrapper`:  Wrapper class for adapting ensemble model inputs/outputs for LAMMPS
- `LAMMPSModelWrapper`: Wrapper class for adapting model inputs/outputs for LAMMPS
- `convert_model_for_lammps`: Function to convert trained model to TorchScript format
- `convert_models_for_lammps`: Function to convert trained ensemble models to TorchScript format

### C++ LAMMPS Plugins
C++ plugins for LAMMPS molecular dynamics simulations:
- `PairIANN`: Single GPU pair style for IANN potentials
- `PairIANNMultiGPU`: Multiple GPU pair style for IANN potentials
- `ComputeIANNVariance`: Compute style for variance calculations


## Troubleshooting

- **Memory Issues**: Reduce batch size or model size if you encounter OOM errors
- **Training Instability**: Try reducing learning rate or using gradient clipping
- **Poor Performance**: try increasing model capacity

## Issues

For questions, issues, and contributions, please use the GitHub issue tracker

## Maintainer
Maintainer `Dr. Changzhi Ai` (changzhi@stanford.edu) at SUNCAT center, Stanford University and SLAC, who is supervised by Dr. Johannes Voss and Dr. Frank Abild-Pedersen.

## References

[1] K. T. Schütt, et al. "Equivariant message passing for the prediction of tensorial properties and molecular spectra", arXiv:2102.03150 (2021). [Link](https://arxiv.org/abs/2102.03150) 

[2] S. Batzner, et al. "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials", Nature Communications, 13, 2453 (2022). [Link](https://doi.org/10.1038/s41467-022-29939-5)

[3] I. Batatia, et al. "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields", arXiv:2206.07697 (2022). [Link](https://arxiv.org/abs/2206.07697)

[4] Y. L. Liao, et al. "EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations", arXiv:2306.12059 (2023). [Link](https://arxiv.org/abs/2306.12059)

[5] X. Yang, et al. "CURATOR: Building Robust Machine Learning Potentials for Atomistic Simulations Autonomously with Batch Active Learning", ChemRxiv (2024). [Link](http://dx.doi.org/10.26434/chemrxiv-2024-p5t3l) 

