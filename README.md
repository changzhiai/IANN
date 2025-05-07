# IANN: InterAtomic Neural Network

## 1. Introductions

IANN (InterAtomic Neural Network) is a equivariant interatomic neural network potential package for materials science and computational chemistry. It implements state-of-the-art graph neural network models for periodic and non-periodic systems, including PaiNN, Nequip, MACE, and EquiformerV2, focusing on predicting energies and forces with high accuracy. 

Key features:
- Multiple equivariant interatomic neural network models implementation
- High-accuracy energy and force predictions
- Distributed training on multiple GPUs and multiple server nodes
- Integration with ASE and LAMMPS for molecular dynamics simulations
- Customizable model architectures

### Documentation
A documentation is available at: https://iann.readthedocs.io

## 2. Installation

### Prerequisites

- ASE (Atomic Simulation Environment) 3.24.0
- PyTorch 1.9+
- Python 3.7+



### Installing IANN

```bash
# Clone the repository
git clone https://github.com/changzhiai/IANN.git
cd IANN

# Install with pip
pip install -e .

# Or install with requirements.txt
pip install -r requirements.txt
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
- Creating and training a PaiNN model
- Evaluating the model
- Using the model for predictions

Check out the `examples/` directory for more sample scripts and tutorials.

## 4. Training

### Preparing your dataset

IANN works with ASE database (.db) or trajectory (.traj) files. Ensure your data contains atomic structures with energy and force labels.

### Training
Create `train.py`
```
from iann.trainer.trainer import Trainer

trainer = Trainer(
    model="painn",
    config={"device": "cpu", 
            'output_dir': 'output'},
    distributed=False
    )
trainer.train("dataset.traj")
```

Available models for `model`:
```
- painn
- nequip
- mace
- equiformerV2
```

Available configurations for `config`:
```
max_steps = 50000
node_size = 128
num_interactions = 3
cutoff = 4.0
val_ratio = 0.1
output_dir = "model_output"
dataset = "path/to/your/data.traj"
batch_size = 32
initial_lr = 0.0001
forces_weight = 0.9
log_interval = 2000
normalization = true
stop_patience = 50
random_seed = 666
load_model = None
device = 'cpu'
```

### Monitoring Training Progress

Training logs will be saved in the specified output directory. You can monitor:
- Energy and force prediction errors
- Training and validation losses
- Model checkpoints

## 5. Predicting

### Loading a Trained Model

```python
from iann.models.painn import PainnModel
import torch

# Load model
state_dict = torch.load("model_output/best_model.pth")
model = PainnModel(
    num_interactions=state_dict["num_layer"],
    hidden_state_size=state_dict["node_size"],
    cutoff=state_dict["cutoff"],
    compute_forces=True
)
model.load_state_dict(state_dict["model"])
```

### Making Predictions with ASE

```python
from iann.calculators.calculators import MLCalculator
from ase.io import read

# Create calculator
calc = MLCalculator(model)

atoms = read("test_structures.traj", ":")
for atom in atoms:
    atom.calc = calc
    energy = atom.get_potential_energy()
    forces = atom.get_forces()
    print(f"Energy: {energy} eV")
    print(f"Forces: {forces} eV/Å")
```
Note: `EnsembleCalculator` and `AtomicEnsembleCalculator` are available to get uncertainty for each structure and each atom, seperately.

## 6. Parallelization

IANN supports distributed training using PyTorch's Distributed Data Parallel (DDP).

### Multi-GPU Training
Submit to multiple GPUs (in SLURM Workload Manager)
```bash
# Run on multiple GPUs and multiple nodes
#!/bin/bash
#SBATCH -N 2                   # Number of nodes
#SBATCH -C gpu                 # Use GPU nodes
#SBATCH -q debug
#SBATCH -t 00:30:00 
#SBATCH --gpus-per-node=4      # Number of GPUs per node
module load pytorch
srun python run.py
```
### Multi-GPU Training
Submit to multiple CPUs (in SLURM Workload Manager)
```bash
# Run on multiple CPUs and multiple nodes
#!/bin/bash
#SBATCH -N 2                   # Number of nodes
#SBATCH -C cpu                 # Use CPU nodes
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node=128  # Number of CPU cores per node
module load pytorch
srun python run.py
```

Note: the parallelization parameters are automatically obtained from the SLURM environment variables.

### Performance Considerations

- Use the largest batch size that fits in your GPU memory
- Enable mixed precision training for faster performance
- Monitor GPU utilization to ensure efficient resource use

## 7. LAMMPS Interface

IANN models can be used as interatomic potentials in LAMMPS molecular dynamics simulations.

### Exporting Models

```python
from iann.plugins.converter import convert_model_for_lammps

convert_model_for_lammps(model_path='best_model.pth', 
                         model_type='painn', 
                         output_path='output_model.pth')
```

### Using with LAMMPS

```
# LAMMPS input script example
units metal
atom_style atomic
boundary p p p

read_data initial.data

# Define the IANN pair style
pair_style iann painn output_model.pt 5.5
pair_coeff * *

mass 1 1.0079999997406976 # H
mass 2 195.08399994981576 # Pt

neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes

# Thermodynamic settings
thermo 10

# Initial minimization to relax the system before dynamics
minimize 1.0e-4 1.0e-6 100 1000

# Run your simulation
timestep 0.001
fix 1 all nvt temp 300.0 300.0 0.1
dump 1 all custom 10 dump.xyz id type x y z

run 2000
```

## 8. Modules

IANN is organized into several key modules:

### iann.data
Data handling utilities:
- `AtomsData`: Data object for each atoms
- `AseDataset`: Dataset class for handling atomic structures
  
### iann.models
Contains neural network model implementations:
- `PaiNN`: PaiNN model implementation for energy and force prediction
- `Nequip`: Nequip model implementation for energy and force prediction
- `MACE`: MACE model implementation for energy and force prediction
- `EquiformerV2`: EquiformerV2 model implementation for energy and force prediction



### iann.calculators
Utility functions:
- `MLCalculator`: ASE calculator interface for models
- `EnsembleCalculator`: ASE ensemble calculator interface for models
- `AtomicEnsembleCalculator`: ASE atomic ensemble calculator interface for models

### iann.plugins
Tools for converting models:
- Convert to LAMMPS

## Troubleshooting

- **Memory Issues**: Reduce batch size or model size if you encounter OOM errors
- **Training Instability**: Try reducing learning rate or using gradient clipping
- **Poor Performance**: try increasing model capacity

## Issues

For questions, issues, and contributions, please use the GitHub issue tracker

## Maintainer
Maintainer `Dr. Changzhi Ai` (changzhi@stanford.edu) at SUNCAT center, Stanford University and SLAC, who is supervised by Dr. Johannes Voss and Dr. Frank Abild-Pedersen .

## References

[1] K. T. Schütt, et al. "Equivariant message passing for the prediction of tensorial properties and molecular spectra", arXiv:2102.03150 (2021). [Link](https://arxiv.org/abs/2102.03150) 

[2] S. Batzner, et al. "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials", Nature Communications, 13, 2453 (2022). [Link](https://doi.org/10.1038/s41467-022-29939-5)

[3] I. Batatia, et al. "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields", arXiv:2206.07697 (2022). [Link](https://arxiv.org/abs/2206.07697)

[4] Y. L. Liao, et al. "EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations", arXiv:2306.12059 (2023). [Link](https://arxiv.org/abs/2306.12059)

[5] X. Yang, et al. "CURATOR: Building Robust Machine Learning Potentials for Atomistic Simulations Autonomously with Batch Active Learning", ChemRxiv (2024). [Link](http://dx.doi.org/10.26434/chemrxiv-2024-p5t3l) 

