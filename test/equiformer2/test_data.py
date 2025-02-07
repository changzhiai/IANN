from fairchem.core.preprocessing import AtomsToGraphs
from fairchem.core.datasets import LmdbDataset
# from ocpmodels.preprocessing.atoms_to_graphs import AtomsToGraphs
# from ocpmodels.datasets.lmdb_dataset import LmdbDataset
import ase.io
from ase.build import bulk
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
import matplotlib.pyplot as plt
import lmdb
import pickle
from tqdm import tqdm
import torch
import os
from fairchem.core.models.base import GraphModelMixin
from fairchem.core.datasets import data_list_collater

# db = lmdb.open(
#     "Pt_all.lmdb",
#     map_size=1099511627776 * 2,
#     subdir=False,
#     meminit=False,
#     map_async=True,
# )

a2g = AtomsToGraphs(
    max_neigh=50,
    radius=6,
    r_energy=True,    # False for test data
    r_forces=True,    # False for test data
    r_distances=False,
    r_fixed=True,
)

raw_data = ase.io.read("/Users/changzhi/Library/CloudStorage/OneDrive-SLACNationalAcceleratorLaboratory/SLAC/IANN/test/equiformer2/Pt_ads.traj", ":")
tags = raw_data[0].get_tags()
data_objects = a2g.convert_all(raw_data, disable_tqdm=True)
inputs = data_list_collater(data_objects)
print(data_objects)


class CustomGraphModel(torch.nn.Module, GraphModelMixin):
    def __init__(self, use_pbc=True, cutoff=5.0, max_neighbors=50):
        super().__init__()
        # self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.use_pbc = False
        self.use_pbc_single = False
        self.otf_graph = False

    def forward(self, data):
        # Generate graph from the ASE Atoms object
        graph = self.generate_graph(data)
        return graph
    
model = CustomGraphModel()
data = model.generate_graph(inputs)
print(data)
# (
#     edge_index,
#     edge_distance,
#     edge_distance_vec,
#     cell_offsets,
#     _,  # cell offset distances
#     neighbors,
# ) = self.generate_graph(data_objects)
# for fid, data in tqdm(enumerate(data_objects), total=len(data_objects)):
#     #assign sid
#     data.sid = torch.LongTensor([0])

#     #assign fid
#     data.fid = torch.LongTensor([fid])

#     #assign tags, if available
#     data.tags = torch.LongTensor(tags)

#     # Filter data if necessary
#     # FAIRChem filters adsorption energies > |10| eV and forces > |50| eV/A

#     # no neighbor edge case check
#     if data.edge_index.shape[1] == 0:
#         print("no neighbors")
#         # print("no neighbors", traj_path)
#         continue

#     txn = db.begin(write=True)
#     txn.put(f"{fid}".encode("ascii"), pickle.dumps(data, protocol=-1))
#     txn.commit()

# txn = db.begin(write=True)
# txn.put(f"length".encode("ascii"), pickle.dumps(len(data_objects), protocol=-1))
# txn.commit()


# db.sync()
# db.close()


