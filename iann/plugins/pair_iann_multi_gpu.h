/* -*- c++ -*- ----------------------------------------------------------
   GPU version of IANN LAMMPS interface
   Inherits from LAMMPS Pair class
   Uses LibTorch C++ API to interface with trained models on GPU(s)
   Works with single GPU or multiple GPU setups
   Handles neighbor list construction, energy/force calculations
   Handles ensemble statistics (energy variance, force variances)
   Handles per-atom energies and variances (if model provides them)
   Handles virial calculation

   Model Input (AtomsData NamedTuple):
     num_atoms        : Tensor[1]    number of atoms (N)
     atomic_numbers   : Tensor[N]    element types (0-indexed)
     positions        : Tensor[N,3]  atomic positions
     cell             : Tensor[3,3]  cell box vectors
     edge_indices     : Tensor[M,2]  edge indices (source, target)
     edge_vectors     : Tensor[M,3]  edge vectors for each edge
     num_edges        : Tensor[1]    total number of neighbor edges (M)
   
   Model Output:
     energy    : Tensor[1]  total system energy
     forces    : Tensor[N,3] per-atom forces
     energy_variance: Tensor[1]  global energy variance
     forces_variance: Tensor[N,3] per-atom force variances
     atomic_energy: Tensor[N] per-atom energies
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(iann/multi_gpu,PairIANNMultiGPU);
// clang-format on
#else

#ifndef LMP_PAIR_IANN_MULTI_GPU_H
#define LMP_PAIR_IANN_MULTI_GPU_H

#include "pair.h"

// Include the Libtorch C++ API
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>
#include <vector>
#include <memory>
#include <thread>
#include <future>

namespace LAMMPS_NS {

class PairIANNMultiGPU : public Pair {
 public:
  PairIANNMultiGPU(class LAMMPS *);
  ~PairIANNMultiGPU() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  
  // Variables for storing ensemble statistics
  double energy_variance;           // Global energy variance
  double force_variance;           // Variance of force magnitudes
  double max_energy_variance;      // Maximum of atomic energy variances
  double max_force_variance;       // Maximum of force variances
  
  // Global properties for thermo output
  double *global_properties;       // Array to store global properties
  int n_global_properties;         // Number of global properties
  
 protected:
  void allocate();
  
  // Model type and path
  char *model_type;      // Type of model (painn, nequip, mace, equiformer2)
  char *model_path;      // Path to the serialized model file
  double cutoff;         // Interaction cutoff distance
  int num_types;         // Number of atom types in the simulation
  
  // Multi-GPU management
  int num_gpus;                    // Number of available GPUs
  std::vector<torch::Device> devices;  // Available GPU devices
  std::vector<std::shared_ptr<torch::jit::Module>> models;  // Models on each GPU
  std::vector<std::thread> worker_threads;  // Worker threads for each GPU
  std::vector<std::promise<torch::Tensor>> energy_promises;  // Promises for energy results
  std::vector<std::promise<torch::Tensor>> forces_promises;  // Promises for forces results
  std::vector<std::promise<torch::Tensor>> energy_var_promises;  // Promises for energy variance
  std::vector<std::promise<torch::Tensor>> forces_var_promises;  // Promises for forces variance
  std::vector<std::promise<torch::Tensor>> atomic_energy_var_promises;  // Promises for atomic energy variance
  
  // Data distribution
  std::vector<int> gpu_assignments;  // Which GPU each atom is assigned to
  std::vector<std::vector<int>> atom_groups;  // Groups of atoms for each GPU
  std::vector<std::vector<int>> edge_groups;  // Groups of edges for each GPU
  
  // LibTorch tensors for ML model input/output (per GPU)
  std::vector<torch::Tensor> num_atoms_tensors;       // Number of atoms per GPU
  std::vector<torch::Tensor> atomic_numbers_tensors;  // Atom types per GPU
  std::vector<torch::Tensor> positions_tensors;       // Atom positions per GPU
  std::vector<torch::Tensor> cell_tensors;            // Unit cell per GPU
  std::vector<torch::Tensor> edge_indices_tensors;    // Neighbor pairs per GPU
  std::vector<torch::Tensor> edge_vectors_tensors;    // Neighbor offsets per GPU
  std::vector<torch::Tensor> num_edges_tensors;       // Number of neighbor pairs per GPU
  
  // LibTorch tensors for single-GPU/CPU fallback (rebuilt each compute)
  torch::Tensor edge_indices_tensor;      // Neighbor pairs (M, 2)
  torch::Tensor edge_vectors_tensor;      // Neighbor offsets (M, 3)
  torch::Tensor num_edges_tensor;         // Number of neighbor pairs (scalar)
  
  // Map from atom tags to sequential indices
  std::vector<int> tag_to_index;
  
  // Compact subgraph mapping
  std::map<int, int> compact_map;  // old_index -> compact_index
  std::vector<int> compact_atoms;  // compact_index -> old_index
  
  // GPU support
  bool use_gpu;          // Whether to use GPU
  bool use_multi_gpu;    // Whether to use multiple GPUs
  
  void build_edges(int inum, int *ilist, int *numneigh, int **firstneigh);
  void distribute_atoms_to_gpus();
  void distribute_edges_to_gpus();
  void run_inference_on_gpu(int gpu_id, int atom_start, int atom_end, int edge_start, int edge_end);
  void collect_results_from_gpus();
  
  bool debug;  // Debug flag
};

}    // namespace LAMMPS_NS

#endif
#endif
