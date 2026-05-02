from iann.trainer import Trainer
import sys
import os

# Add IANN root to path if running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

trainer = Trainer(
   model="allegro",
   config={"num_channels": 64, # number of channels for chemical embedding
           "num_scalar_features": 64, # size of scalar features
           "num_tensor_features": 16, # size of tensor features
           "num_layers": 2, # number of allegro interaction layers
           "lmax": 1, # max L for spherical harmonics
           "cutoff": 5.5, # cutoff radius
           "batch_size": 16, # batch size
           "learning_rate": 0.001, # initial learning rate
           "forces_weight": 0.99, # weight for forces
           # "load_model": 'output/model.pt', # load model from checkpoint
           "max_steps": 30000000, # maximum number of steps.
           "random_seed": 889, # random seed for reproducibility
           "val_ratio": 0.003, # validation ratio
           "stop_patience": 600, # patience for early stopping
           "log_interval": 1,
           "norm_data": True, # global data normalization
           "norm_per_atom": True, # normalize per atom
           "use_cue": False, # cuEquivariance (set to True if available)
           "device": "cpu", # run on CPU for basic testing
           "output_dir": "test/allegro/output",
           "output_log": "output.log",
           "output_model": "model.pt"},
   distributed=False,
)

# Run training on a small test dataset
trainer.train("test/Pt_ads.traj")
