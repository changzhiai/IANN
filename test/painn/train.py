from iann.trainer.trainer import Trainer
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


trainer = Trainer(
    model="painn",
    config={"device": "cpu",
            "node_size": 128,
            'output_dir': 'test/painn/model_output',
            "log_interval": 1, },
    distributed=False
    )
trainer.train("test/Pt_ads.traj")