from iann.trainer import Trainer
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

trainer = Trainer(
   model="mace",
   config={"num_channels": 128, # number of channels in the model
           "num_layers": 2, # number of layers in the model
           "lmax": 1, # 128x0e + 128x1o
           "cutoff": 5.5, # cutoff radius
           "batch_size": 16, # batch size
           "learning_rate": 0.0001, # initial learning rate
           "forces_weight": 0.9, # weight for forces
           # "load_model": 'test/mace/output', # load model from checkpoint
           "max_steps": 30000000, # maximum number of steps
           "random_seed": 666, # random seed for reproducibility
           "val_ratio": 0.002, # validation ratio
           "stop_patience": 600, # patience for early stopping
           'log_interval': 1,
           'norm_data': True, # normalize data
           'norm_per_atom': True, # normalize per atom
           'use_cue': False, # use cue
           # "max_epochs": 5,
           'device': 'cpu',
           'output_dir': 'test/mace/output',
           'output_log': 'output.log',
           'output_model': 'model.pt'},
   distributed=False,
   )

trainer.train("test/Pt_ads.traj")