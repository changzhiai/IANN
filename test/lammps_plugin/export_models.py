from iann.plugins.converter import convert_model_for_lammps
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# convert_model_for_lammps(model_path='test/mace/model_output/best_model.pth', 
#                          model_type='mace', 
#                          output_path='test/lammps_plugin/export_mace.pth')

convert_model_for_lammps(model_path='test/nequip/model_output/best_model.pth', 
                         model_type='nequip', 
                         output_path='test/lammps_plugin/export_nequip.pth')

# convert_model_for_lammps(model_path='test/painn/model_output/best_model.pth', 
#                          model_type='painn', 
#                          output_path='test/lammps_plugin/export_painn.pth')


