from iann.trainer.trainer import Trainer
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


trainer = Trainer(
    model="nequip",
    config={"device": "cpu", 
            'output_dir': 'test/nequip/model_output'},
    distributed=False
    )
trainer.train("test/Pt_ads.traj")