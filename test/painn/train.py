from iann.trainer import Trainer
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

trainer = Trainer(
   model="painn",
   config={"num_channels": 128, # number of channels in the model
            "num_layers": 3, # number of layers in the model
            "cutoff": 5.5, # cutoff radius
            "batch_size": 16, # batch size
            "learning_rate": 0.0001, # initial learning rate
            "forces_weight": 0.9, # weight for forces
            "load_model": True, # load model from checkpoint
            "max_steps": 30000000, # maximum number of steps
            "random_seed": 888, # random seed for reproducibility
            "val_ratio": 0.003, # validation ratio
            "stop_patience": 600, # patience for early stopping
            "log_interval": 1, # log interval
            # "max_epochs": 5,
            'device': 'cuda',
            'output_dir': 'test/painn/output',
            'output_log': 'output.log',
            'output_model': 'model.pt'},
   distributed=True,
   )


# trainer = Trainer(
#     model="painn",
#     config={"device": "cpu",
#             "num_channels": 64,
#             "num_layers": 3,
#             'output_dir': 'test/painn/output',
#             # 'load_model': 'test/painn/output/model.pt',
#             "log_interval": 1,
#             "learning_rate": 0.0001,
#             "stop_patience": 400,
#             'forces_weight': 0.999,
#             },
#     distributed=False
#     )
trainer.train("test/Pt_ads.traj")
# trainer.train("test/dft_PdTiH_adss_r0_to_r31_final_tot.traj")