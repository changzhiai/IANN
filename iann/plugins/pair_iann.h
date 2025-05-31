/* -*- c++ -*- ----------------------------------------------------------
   Inherits from LAMMPS Pair class
   Uses LibTorch C++ API to interface with trained models
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
PairStyle(iann,PairIANN);
// clang-format on
#else

#ifndef LMP_PAIR_IANN_H
#define LMP_PAIR_IANN_H

#include "pair.h"

// Include the Libtorch C++ API
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>

namespace LAMMPS_NS {

class PairIANN : public Pair {
 public:
  PairIANN(class LAMMPS *);
  ~PairIANN() override;
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
  
  // LibTorch tensors for ML model input/output
  torch::Tensor num_atoms_tensor;       // Number of atoms (scalar)
  torch::Tensor atomic_numbers_tensor;  // Atom types (N)
  torch::Tensor positions_tensor;     // Atom positions (N, 3)
  torch::Tensor cell_tensor;            // Unit cell (3, 3)
  torch::Tensor edge_indices_tensor;      // Neighbor pairs (M, 2)
  torch::Tensor edge_vectors_tensor;      // Neighbor offsets (M, 3)
  torch::Tensor num_edges_tensor;         // Number of neighbor pairs (scalar)
  
  torch::Tensor neighbor_offsets; // Neighbor offsets (M, 3)
  torch::Tensor neighbor_indices;
  
  // Map from atom tags to sequential indices
  std::vector<int> tag_to_index;
  
  // Loaded TorchScript model
  std::shared_ptr<torch::jit::Module> model;
  
  // GPU support
  bool use_gpu;          // Whether to use GPU
  torch::Device device;  // Device to run model on (CPU or CUDA)
  
  void build_edges(int inum, int *ilist, int *numneigh, int **firstneigh);
  
  bool debug;  // Debug flag
};

}    // namespace LAMMPS_NS

#endif
#endif 