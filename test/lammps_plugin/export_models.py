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

    elif export == 'allegro':
        convert_model_for_lammps(model_path='test/allegro/output/model.pt',
                                model_type='allegro',
                                output_path='test/lammps_plugin/export_allegro.pt', **kwargs)

    elif export == 'equiformerV3':
        convert_model_for_lammps(model_path='test/equiformerV3/output/model.pt',
                                model_type='equiformerV3',
                                output_path='test/lammps_plugin/export_equiformerV3.pt', **kwargs)

    elif export == 'uma':
        convert_model_for_lammps(model_path='test/uma/output/model.pt',
                                model_type='uma',
                                output_path='test/lammps_plugin/export_uma.pt', **kwargs)

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

def _try(name, **kwargs):
    """Run one export and record the outcome.

    Each architecture is attempted independently: aborting on the first failure
    used to hide every later one, which is how the EquiformerV2 breakage masked
    the fact that allegro/equiformerV3/uma were never being exercised here.
    """
    try:
        export_models(export=name, **kwargs)
        _RESULTS.append((name, "OK"))
    except Exception as exc:                       # noqa: BLE001 - reporting only
        first = (str(exc).strip().splitlines() or [""])[0][:70]
        _RESULTS.append((name, f"FAIL {type(exc).__name__}: {first}"))


_RESULTS = []

if __name__ == "__main__":
    _try("painn")
    _try("nequip",
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
    _try("mace",
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
    _try("equiformerV2",
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
    # The three architectures added after the original four. Structural
    # parameters must be repeated here because the trainer does not persist all
    # of them in the checkpoint (num_distance_basis, the grid resolution lists),
    # so reconstruction would otherwise fall back to defaults and the
    # state_dict would not fit.
    _try("allegro",
                    num_channels=64,
                    num_scalar_features=64,
                    num_tensor_features=16,
                    num_layers=2,
                    lmax=1,
                    cutoff=5.5,
                    use_cue=False,
                    norm_data=True,
                    norm_per_atom=True,
                    device='cpu',
                    output_dir='test/allegro/output',
                    output_log='output.log',
                    output_model='model.pt')
    _try("equiformerV3",
                    num_layers=2,
                    num_channels=16,
                    lmax=3,
                    mmax=2,
                    attn_grid_resolution_list=[12, 6],
                    ffn_grid_resolution_list=[12, 12],
                    norm_type='merge_layer_norm',
                    attn_activation='sep-merge_gates2_swiglu',
                    ffn_activation='sep-merge_gates2_swiglu',
                    use_envelope=True,
                    norm_data=True,
                    norm_per_atom=True,
                    device='cpu',
                    output_dir='test/equiformerV3/output',
                    output_log='output.log',
                    output_model='model.pt')
    _try("uma",
                    num_channels=32,
                    num_layers=2,
                    lmax=2,
                    mmax=2,
                    hidden_channels=32,
                    edge_channels=32,
                    num_distance_basis=128,
                    cutoff=5.5,
                    norm_type='rms_norm_sh',
                    norm_data=True,
                    norm_per_atom=True,
                    device='cpu',
                    output_dir='test/uma/output',
                    output_log='output.log',
                    output_model='model.pt')
    _try("ensemble_painn")

    print("\n==================== export summary ====================")
    for _name, _status in _RESULTS:
        print(f"  {_name:16s} {_status}")
    _failed = [n for n, st in _RESULTS if st != "OK"]
    print(f"  {len(_RESULTS) - len(_failed)}/{len(_RESULTS)} exported")
    raise SystemExit(1 if _failed else 0)
