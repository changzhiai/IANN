# IANN: InterAtomic Neural Network

## 1. About

IANN (InterAtomic Neural Network) is a PyTorch-based package for machine learning in materials science and computational chemistry. It implements state-of-the-art graph neural network models for atomic systems, focusing on predicting energies and forces with high accuracy.

Key features:
- PaiNN (Polarizable Atom-Interaction Neural Network) model implementation
- High-accuracy energy and force predictions
- Distributed training on multiple GPUs
- Integration with ASE and LAMMPS for molecular dynamics simulations
- Customizable model architectures

## 2. Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.9+
- ASE (Atomic Simulation Environment)

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

### Configuration

Create a TOML configuration file with your training parameters:

```toml
node_size = 128
num_interactions = 3
cutoff = 4.0
val_ratio = 0.1
output_dir = "model_output"
dataset = "path/to/your/data.traj"
max_steps = 100000
batch_size = 32
initial_lr = 0.0001
forces_weight = 0.9
log_interval = 2000
normalization = true
atomwise_normalization = true
stop_patience = 50
random_seed = 42
```

### Running Training

```bash
# Single-GPU training
python test/painn/train.py --cfg config.toml
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
from iann.utils import MLCalculator
from ase.io import read

# Create calculator
calc = MLCalculator(model)

# Read structures and make predictions
atoms = read("test_structures.traj", ":")
for atom in atoms:
    atom.calc = calc
    energy = atom.get_potential_energy()
    forces = atom.get_forces()
    print(f"Energy: {energy} eV")
    print(f"Forces: {forces} eV/Å")
```

## 6. Parallelization

IANN supports distributed training using PyTorch's Distributed Data Parallel (DDP).

### Multi-GPU Training

```bash
# Run on multiple GPUs
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS test/painn/train.py --cfg config.toml
```

Note: When using DDP, you may see a warning about gradient strides not matching bucket view strides. This is not an error and typically doesn't affect training accuracy.

### Performance Considerations

- Use the largest batch size that fits in your GPU memory
- Enable mixed precision training for faster performance
- Monitor GPU utilization to ensure efficient resource use

## 7. LAMMPS Interface

IANN models can be used as interatomic potentials in LAMMPS molecular dynamics simulations.

### Exporting Models

```python
from iann.export import export_lammps
export_lammps(model, "model_lammps.pt")
```

### Using with LAMMPS

```
# LAMMPS input script example
units metal
atom_style atomic

# Load ML potential
pair_style mlip
pair_coeff * * model_lammps.pt

# Run MD
fix 1 all nve
timestep 0.001
run 10000
```

## 8. Modules

IANN is organized into several key modules:

### iann.models
Contains neural network model implementations:
- `PainnModel`: Main model implementation for energy and force prediction
- `PainnMessage`: Message passing implementation
- `PainnUpdate`: Node update implementation

### iann.data
Data handling utilities:
- `AseDataset`: Dataset class for handling atomic structures
- `collate_atomsdata`: Collate function for batching atomic data

### iann.utils
Utility functions:
- `MLCalculator`: ASE calculator interface for models
- Normalization and cutoff functions

### iann.export
Tools for exporting and converting models:
- Export to LAMMPS
- Export to ONNX

## Troubleshooting

- **Memory Issues**: Reduce batch size or model size if you encounter OOM errors
- **Training Instability**: Try reducing learning rate or using gradient clipping
- **Poor Performance**: Ensure your dataset is properly normalized and try increasing model capacity

## Contact and Support

For questions, issues, and contributions, please use the GitHub issue tracker or contact the maintainers directly.



``` 