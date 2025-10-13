from iann.trainer import Trainer
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


trainer = Trainer(
    model="nequip",
    config={"device": "cpu",
            'num_channels': 64,
            'output_dir': 'test/nequip/model_output',
            # 'load_model': 'test/nequip/model_output/model.pt',
            'log_interval': 1},
    distributed=False
    )
trainer.train("test/Pt_ads.traj")