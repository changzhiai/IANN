from iann.trainer import Trainer
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_channels", type=int, default=128)
parser.add_argument("--output_dir", type=str, default='test/painn/output')
args = parser.parse_args()

trainer = Trainer(
   model="painn",
   config={"num_channels": args.num_channels, # number of channels in the model
            "num_layers": 3, # number of layers in the model
            "cutoff": 5.5, # cutoff radius
            "batch_size": 16, # batch size
            "learning_rate": 0.0001, # initial learning rate
            "forces_weight": 0.9, # weight for forces
            # "load_model": 'test/painn/output/model.pt', # load model from checkpoint
            "max_steps": 30000000, # maximum number of steps
            "random_seed": 888, # random seed for reproducibility
            "val_ratio": 0.003, # validation ratio
            "stop_patience": 600, # patience for early stopping
            "log_interval": 1, # log interval
            # "max_epochs": 5,
            'device': 'cuda',
            'output_dir': args.output_dir,
            'output_log': 'output.log',
            'output_model': 'model.pt'},
   distributed=False,
   )

trainer.train("test/Pt_ads.traj")
# trainer.train("test/dft_PdTiH_adss_r0_to_r31_final_tot.traj")