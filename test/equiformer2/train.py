from iann.trainer.trainer import Trainer
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


trainer = Trainer(
    model="equiformerV2",
    config={"device": "cpu", 
            'output_dir': 'test/equiformerV2/model_output'},
    distributed=False
    )
trainer.train("test/Pt_ads.traj")