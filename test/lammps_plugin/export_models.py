from iann.plugins.converter import convert_model_for_lammps, convert_models_for_lammps
import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def export_models(export="painn"):
    if export == 'equiformerV2':
        convert_model_for_lammps(model_path='test/equiformerV2/model_output/best_model.pth', 
                                model_type='equiformerV2', 
                                output_path='test/lammps_plugin/export_equiformerV2.pth')

    elif export == 'mace':
        convert_model_for_lammps(model_path='test/mace/model_output/best_model.pth', 
                                model_type='mace', 
                                output_path='test/lammps_plugin/export_mace.pth')

    elif export == 'nequip':
        convert_model_for_lammps(model_path='test/nequip/model_output/best_model.pth', 
                                model_type='nequip', 
                                output_path='test/lammps_plugin/export_nequip.pth')

    elif export == 'painn':
        convert_model_for_lammps(model_path='test/painn/model_output/best_model.pth', 
                                model_type='painn', 
                                output_path='test/lammps_plugin/export_painn.pth')

    elif export == 'ensemble_painn':
        model_paths = [
            "test/painn/model_output_124/best_model.pth",
            "test/painn/model_output_128/best_model.pth",
            "test/painn/model_output_132/best_model.pth"
        ]
        output_path = convert_models_for_lammps(
            model_paths=model_paths,
            model_type="painn",
            output_path="test/lammps_plugin/export_ensemble_painn.pth"
        )

if __name__ == "__main__":
    export_models(export="equiformerV2")