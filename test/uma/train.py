from iann.trainer import Trainer
import sys
import os

# Add IANN root to path if running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

trainer = Trainer(
    model="uma",
    config={
        "num_channels": 32,        # sphere_channels (small for testing)
        "num_layers": 2,           # number of eSCNMD blocks
        "lmax": 2,                 # max spherical harmonic degree
        "mmax": 2,                 # max order for SO(2) conv
        "hidden_channels": 32,     # hidden channels per block
        "edge_channels": 32,       # edge embedding channels
        "num_distance_basis": 128, # Gaussian basis functions
        "cutoff": 5.5,             # cutoff radius
        "norm_type": "rms_norm_sh",
        "batch_size": 12,
        "learning_rate": 0.001,
        "forces_weight": 0.7,
        "log_interval": 1,
        "max_steps": 30000000,
        "stop_patience": 1000,
        "norm_data": True,
        "norm_per_atom": True,
        "device": "cpu",
        "output_dir": "test/uma/output",
        "output_log": "output.log",
        "output_model": "model.pt",
    },
    distributed=False,
)

# Run training on a small test dataset
trainer.train("test/Pt_ads.traj")
