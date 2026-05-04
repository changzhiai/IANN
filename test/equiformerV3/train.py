from iann.trainer import Trainer
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


trainer = Trainer(
    model="equiformerV3",
    config={"device": "cpu",
            'output_dir': 'test/equiformerV3/output',
            'num_layers': 2,
            'num_channels': 16,
            'batch_size': 8,
            'forces_weight': 0.7,
            'log_interval': 1,
            'max_grad_norm': 1.0,
            'learning_rate': 0.001,
            'lmax': 3,
            'mmax': 2,
            'attn_grid_resolution_list': [12, 6],
            'ffn_grid_resolution_list': [12, 12],
            'norm_type': 'merge_layer_norm',
            'attn_activation': 'sep-merge_gates2_swiglu',
            'ffn_activation': 'sep-merge_gates2_swiglu',
            'use_envelope': True,
            "stop_patience": 1000,
            'norm_data': True,
            'norm_per_atom': True,
            'log_input': True,
            'output_model': 'model.pt',
            # 'load_model': 'test/equiformerV3/output/model.pt',
            },
    distributed=False
    )
trainer.train("test/Pt_ads.traj")
