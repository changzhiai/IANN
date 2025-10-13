from iann.trainer import Trainer
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


trainer = Trainer(
    model="mace",
    config={"device": "cpu", 
            'num_channels': 64,
            'output_dir': 'test/mace/output',
            # 'load_model': 'test/mace/output/model.pt',
            'norm_data': True,
            'log_interval': 1},
    distributed=False
    )
trainer.train("test/Pt_ads.traj")