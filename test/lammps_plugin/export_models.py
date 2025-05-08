from iann.plugins.converter import convert_model_for_lammps
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


convert_model_for_lammps(model_path='test/painn/model_output/best_model.pth', 
                         model_type='painn', 
                         output_path='test/lammps_plugin/export_painn.pth')