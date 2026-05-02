from iann.plugins.converter import convert_model_for_lammps, convert_models_for_lammps
import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def export_models(export="painn", **kwargs):
    if export == 'equiformerV2':
        convert_model_for_lammps(model_path='test/equiformerV2/output/model.pt', 
                                model_type='equiformerV2', 
                                output_path='test/lammps_plugin/export_equiformerV2.pt', **kwargs)

    elif export == 'mace':
        convert_model_for_lammps(model_path='test/mace/output/model.pt', 
                                model_type='mace', 
                                output_path='test/lammps_plugin/export_mace.pt', **kwargs)

    elif export == 'nequip':
        convert_model_for_lammps(model_path='test/nequip/output/model.pt', 
                                model_type='nequip', 
                                output_path='test/lammps_plugin/export_nequip.pt', **kwargs)

    elif export == 'painn':
        convert_model_for_lammps(model_path='test/painn/output/model.pt', 
                                model_type='painn', 
                                output_path='test/lammps_plugin/export_painn.pt', **kwargs)

    elif export == 'ensemble_painn':
        model_paths = [
            "test/painn/output_124/model.pt",
            "test/painn/output_128/model.pt",
            "test/painn/output_132/model.pt"
        ]
        output_path = convert_models_for_lammps(
            model_paths=model_paths,
            model_type="painn",
            output_path="test/lammps_plugin/export_ensemble_painn.pt"
        )

if __name__ == "__main__":
    export_models(export="painn")
    export_models(export="nequip", 
                   num_channels=128, 
                   num_layers=2, 
                   lmax=1, 
                   parity=True,
                   cutoff=5.5, 
                   batch_size=16, 
                   learning_rate=0.001, 
                   forces_weight=0.99, 
                   max_steps=30000000, 
                   random_seed=889, 
                   val_ratio=0.003, 
                   stop_patience=600, 
                   log_interval=1,
                   norm_data=True, 
                   norm_per_atom=True, 
                   use_cue=False,
                   device='cpu',
                   output_dir='test/nequip/output',
                   output_log='output.log',
                   output_model='model.pt')
    export_models(export="mace",
                    num_channels=128, # number of channels in the model
                    num_layers=2, # number of layers in the model
                    lmax=1, # 128x0e + 128x1o
                    cutoff=5.5, # cutoff radius
                    batch_size=16, # batch size
                    learning_rate=0.0001, # initial learning rate
                    forces_weight=0.9, # weight for forces
                    max_steps=30000000, # maximum number of steps
                    random_seed=777, # random seed for reproducibility
                    val_ratio=0.003, # validation ratio
                    stop_patience=600, # patience for early stopping
                    log_interval=1,
                    norm_data=True, # normalize data
                    norm_per_atom=True, # normalize per atom
                    use_cue=True, # use cue
                    device='cpu',
                    output_dir='test/mace/output',
                    output_log='output.log',
                    output_model='model.pt')
    export_models(export="equiformerV2",
                    device = "cpu", 
                    output_dir = 'test/equiformerV2/output',
                    num_layers = 3,
                    num_channels = 8,
                    batch_size = 12,
                    forces_weight = 0.7,
                    log_interval = 1,
                    max_grad_norm = 1.0,
                    learning_rate = 0.001,
                    grid_resolution = 12,
                    lmax_list = [4],
                    mmax_list = [2],
                    stop_patience = 1000,
                    norm_data = True,
                    norm_per_atom=True,
                    log_input = True,
            output_model = 'model.pt')
    export_models(export="ensemble_painn")