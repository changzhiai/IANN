from iann.trainer import Trainer
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


trainer = Trainer(
    model="equiformerV2",
    config={"device": "cpu", 
            'output_dir': 'test/equiformerV2/output',
            'num_layers': 3,
            'num_channels': 8,
            'batch_size': 12,
            'forces_weight': 0.7,
            'log_interval': 1,
            'max_grad_norm': 1.0,
            'learning_rate': 0.001,
            'grid_resolution': 12,
            'lmax_list': [4],
            'mmax_list': [2],
            "stop_patience": 1000,
            'norm_data': True,
            'norm_per_atom': True,
            'log_input': True,
            'output_model': 'model.pt',
            # 'load_model': 'test/equiformerV2/output/model.pt',
            },
    distributed=False
    )
trainer.train("test/Pt_ads.traj")
