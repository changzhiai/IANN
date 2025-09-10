from iann.trainer import Trainer
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


trainer = Trainer(
    model="painn",
    config={"device": "cpu",
            "num_channels": 64,
            "num_layers": 3,
            'output_dir': 'test/painn/output',
            # 'load_model': 'test/painn/output/model.pt',
            "log_interval": 1,
            "learning_rate": 0.0001,
            "stop_patience": 400,
            'forces_weight': 0.999,
            },
    distributed=False
    )
# trainer.train("test/Pt_ads.traj")
trainer.train("test/dft_PdTiH_adss_r0_to_r31_final_tot.traj")