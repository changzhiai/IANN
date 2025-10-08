/* ----------------------------------------------------------------------
   Multi-GPU version of IANN LAMMPS interface
   Inherits from LAMMPS Pair class
   Uses LibTorch C++ API to interface with trained models across multiple GPUs
   Handles neighbor list construction, energy/force calculations

Due to LAMMPS version, the code is slightly different from the original one.
If you have the following errors:

1. message error
replace all 'message' with 'all', for example:
```
//error->message(FLERR, "[PAIR_IANN] GPU available");
all(FLERR, "[PAIR_IANN] GPU available");
```

2. minimum_image error
replace domain->minimum_image(dx, dy, dz) with domain->minimum_image(std::string(__FILE__), __LINE__, dx, dy, dz) as follows:
```
//domain->minimum_image(dx, dy, dz);
domain->minimum_image(std::string(__FILE__), __LINE__, dx, dy, dz);
```
------------------------------------------------------------------------- */

#include "pair_iann_multi_gpu.h"

#include <cmath>
#include <cstring>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <algorithm>
#include <numeric>
#include <unordered_map>

// LibTorch headers
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>
#include <vector>

// CUDA headers
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
#include <c10/cuda/CUDAGuard.h>

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"
#include "output.h"
#include "thermo.h"
#include "domain.h"
#include <mpi.h>  // for MPI_Allgather, MPI_Allgatherv

using namespace LAMMPS_NS;

std::map<int, int> type_to_Z_global;
std::map<double, int> mass_to_Z_global = {
    {1.008, 1},     // H
    {4.0026, 2},    // He
    {6.94, 3},      // Li
    {9.0122, 4},    // Be
    {10.81, 5},     // B
    {12.011, 6},    // C
    {14.007, 7},    // N
    {15.999, 8},    // O
    {18.998, 9},    // F
    {20.180, 10},   // Ne
    {22.990, 11},   // Na
    {24.305, 12},   // Mg
    {26.982, 13},   // Al
    {28.085, 14},   // Si
    {30.974, 15},   // P
    {32.06, 16},    // S
    {35.45, 17},    // Cl
    {39.948, 18},   // Ar
    {39.098, 19},   // K
    {40.078, 20},   // Ca
    {47.867, 22},   // Ti
    {50.942, 23},   // V
    {51.996, 24},   // Cr
    {54.938, 25},   // Mn
    {55.845, 26},   // Fe
    {58.933, 27},   // Co
    {58.693, 28},   // Ni
    {63.546, 29},   // Cu
    {65.38, 30},    // Zn
    {69.723, 31},   // Ga
    {72.630, 32},   // Ge
    {74.922, 33},   // As
    {78.96, 34},    // Se
    {79.904, 35},   // Br
    {83.798, 36},   // Kr
    {107.868, 47},  // Ag
    {118.71, 50},   // Sn
    {126.904, 53},  // I
    {131.293, 54},  // Xe
    {195.084, 78},  // Pt
    {200.592, 80},  // Hg
    {204.38, 81},   // Tl
    {207.2, 82}     // Pb
};

int get_atomic_number_global(double mass) {
    for (const auto& [m, z] : mass_to_Z_global) {
        if (fabs(mass - m) < 0.1) {  // 0.1 u tolerance
            return z;
        }
    }
    return -1;  // Unknown
}

/* ---------------------------------------------------------------------- */

PairIANNMultiGPU::PairIANNMultiGPU(LAMMPS *lmp) : Pair(lmp),
  use_gpu(false),  // Initialize to false by default
  use_multi_gpu(false),  // Initialize to false by default
  num_gpus(0),
  current_gpu_id(-1)  // Initialize to -1 (no GPU)
{
  restartinfo = 0;
  manybody_flag = 1;
  
  model_type = nullptr;
  model_path = nullptr;
  cutoff = 0.0;
  debug = false;  // Initialize debug flag to false
  
  // Default settings
  comm_forward = 1;  // We need to communicate forces
  comm_reverse = 1;  // We need to gather positions

  // Initialize ensemble statistics variables
  energy_variance = 0.0;
  force_variance = 0.0;
  max_energy_variance = 0.0;
  max_force_variance = 0.0;

  // Initialize global properties
  n_global_properties = 4;  // We have 4 variance values to track
  global_properties = new double[n_global_properties];
  for (int i = 0; i < n_global_properties; i++) {
    global_properties[i] = 0.0;
  }
  
  // Initialize promises for multi-GPU communication
  energy_promises.resize(8);  // Support up to 8 GPUs
  forces_promises.resize(8);
  energy_var_promises.resize(8);
  forces_var_promises.resize(8);
  atomic_energy_var_promises.resize(8);

  // Check CUDA environment and detect available GPUs
  const char* cuda_path = getenv("CUDA_HOME");
  const char* ld_path = getenv("LD_LIBRARY_PATH");
  
  if (comm->me == 0) {
    if (!cuda_path) {
      error->warning(FLERR, "[PAIR_IANN_MULTI_GPU] CUDA_HOME environment variable not set");
    } else {
      // Check CUDA version
      std::string cuda_version;
      std::ifstream version_file(std::string(cuda_path) + "/version.txt");
      if (version_file.is_open()) {
        std::getline(version_file, cuda_version);
        error->message(FLERR, ("[PAIR_IANN_MULTI_GPU] Found CUDA version: " + cuda_version).c_str());
      }
    }
    if (!ld_path) {
      error->warning(FLERR, "[PAIR_IANN_MULTI_GPU] LD_LIBRARY_PATH environment variable not set");
    }
  }

  // Try to initialize CUDA and detect available GPUs
  try {
    if (torch::cuda::is_available()) {
      num_gpus = torch::cuda::device_count();
      if (comm->me == 0) {
        error->message(FLERR, ("[PAIR_IANN_MULTI_GPU] Found " + std::to_string(num_gpus) + " GPU(s)").c_str());
      }
      
      // Initialize device list
      for (int i = 0; i < num_gpus; i++) {
        devices.push_back(torch::Device(torch::kCUDA, i));
      }
      
      // Test each GPU with a small test case
      for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        try {
          torch::Device test_device(torch::kCUDA, gpu_id);
          int ntest = 20;
          torch::Tensor test_positions = torch::zeros({ntest, 3}, torch::kFloat32);
          
          // Try to move tensors to specific GPU
          test_positions = test_positions.to(test_device);
          
          // Perform a simple operation on GPU
          test_positions = test_positions + 1.0;
          
          // Move back to CPU and verify
          test_positions = test_positions.cpu();
          if (test_positions[0][0].item<float>() == 1.0) {
            if (comm->me == 0) {
              error->message(FLERR, ("[PAIR_IANN_MULTI_GPU] GPU " + std::to_string(gpu_id) + " available").c_str());
            }
          }
        } catch (const c10::Error& e) {
          if (comm->me == 0) {
            std::string msg = "[PAIR_IANN_MULTI_GPU] GPU " + std::to_string(gpu_id) + " test failed: ";
            msg += e.what();
            error->warning(FLERR, msg.c_str());
          }
        }
      }
      
      if (num_gpus > 0) {
        use_gpu = true;
        // Always use single GPU mode for simplicity - works for both single and multiple GPU setups
        use_multi_gpu = false;
        if (comm->me == 0) {
          error->message(FLERR, "[PAIR_IANN_MULTI_GPU] GPU mode enabled");
        }
      } else {
        use_gpu = false;
        use_multi_gpu = false;
        if (comm->me == 0) {
          error->message(FLERR, "[PAIR_IANN_MULTI_GPU] No GPUs available, using CPU mode");
        }
      }
    }
  } catch (const c10::Error& e) {
    use_gpu = false;
    use_multi_gpu = false;
    num_gpus = 0;
    if (comm->me == 0) {
      std::string msg = "[PAIR_IANN_MULTI_GPU] GPU test failed: ";
      msg += e.what();
      msg += "\n  Falling back to CPU mode.";
      msg += "\n  Please ensure CUDA libraries are in your LD_LIBRARY_PATH.";
      msg += "\n  You may need to set:";
      msg += "\n  export CUDA_HOME=/path/to/cuda";
      msg += "\n  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH";
      error->warning(FLERR, msg.c_str());
    }
  }

  if (!use_gpu && comm->me == 0) {
    error->message(FLERR, "[PAIR_IANN_MULTI_GPU] Using CPU mode");
  }
}

/* ---------------------------------------------------------------------- */

PairIANNMultiGPU::~PairIANNMultiGPU() 
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
  
  if (model_type) delete[] model_type;
  if (model_path) delete[] model_path;
  
  // Clean up global properties
  delete[] global_properties;
  
  // Clean up GPU resources
  if (use_multi_gpu) {
    // Wait for all worker threads to finish
    for (auto& thread : worker_threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }
  
  // LibTorch cleanup is automatic via shared_ptr
}

/* ---------------------------------------------------------------------- */

void PairIANNMultiGPU::compute(int eflag, int vflag) 
{
  ev_init(eflag, vflag);
  
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int natoms_global = atom->natoms;  // Total number of atoms in the system

  // Build edges based on neighborlist
  build_edges(list->inum, list->ilist, list->numneigh, list->firstneigh);

  // Use compact subgraph - only atoms that appear in edges
  int compact_size = compact_atoms.size();
  torch::Tensor num_atoms_tensor = torch::tensor({compact_size}, torch::kInt64);

  // Convert atom types to atomic numbers for compact subgraph
  std::vector<int> type_to_Z(atom->ntypes + 1); 
  for (int itype = 1; itype <= atom->ntypes; ++itype) {
      double mass = atom->mass[itype];
      type_to_Z[itype] = get_atomic_number_global(mass);
  }
  std::vector<int64_t> atomic_numbers(compact_size);
  for (int i = 0; i < compact_size; ++i) {
      int atom_idx = compact_atoms[i];
      atomic_numbers[i] = type_to_Z[atom->type[atom_idx]];
  }
  torch::Tensor atomic_numbers_tensor = torch::from_blob(atomic_numbers.data(), {compact_size}, torch::TensorOptions().dtype(torch::kInt64)).clone();

  // Convert atom positions to tensor for compact subgraph
  torch::Tensor positions_tensor = torch::zeros({compact_size, 3}, torch::kFloat32);
  auto coord_acc = positions_tensor.accessor<float, 2>();
  for (int i = 0; i < compact_size; i++) {
      int atom_idx = compact_atoms[i];
      coord_acc[i][0] = static_cast<float>(x[atom_idx][0]);
      coord_acc[i][1] = static_cast<float>(x[atom_idx][1]);
      coord_acc[i][2] = static_cast<float>(x[atom_idx][2]);
  }

  // Convert box to tensor (3,3)
  torch::Tensor cell_tensor = torch::zeros({3,3}, torch::kFloat32);
  auto cell_acc = cell_tensor.accessor<float,2>();
  
  double *boxlo = domain->boxlo;
  double *boxhi = domain->boxhi;
  double xy = domain->xy;
  double xz = domain->xz;
  double yz = domain->yz;
  
  cell_acc[0][0] = static_cast<float>(boxhi[0] - boxlo[0]);
  cell_acc[0][1] = 0.0f;
  cell_acc[0][2] = 0.0f;
  cell_acc[1][0] = static_cast<float>(xy);
  cell_acc[1][1] = static_cast<float>(boxhi[1] - boxlo[1]);
  cell_acc[1][2] = 0.0f;
  cell_acc[2][0] = static_cast<float>(xz);
  cell_acc[2][1] = static_cast<float>(yz);
  cell_acc[2][2] = static_cast<float>(boxhi[2] - boxlo[2]);

  // Move tensors to GPU if available
  if (use_gpu) {
    current_gpu_id = 0;  // Set current GPU ID for single GPU mode
    num_atoms_tensor = num_atoms_tensor.to(devices[0]);
    atomic_numbers_tensor = atomic_numbers_tensor.to(devices[0]);
    positions_tensor = positions_tensor.to(devices[0]);
    cell_tensor = cell_tensor.to(devices[0]);
    edge_indices_tensor = edge_indices_tensor.to(devices[0]);
    edge_vectors_tensor = edge_vectors_tensor.to(devices[0]);
    num_edges_tensor = num_edges_tensor.to(devices[0]);
  }

  // Run inference
  try {
    // Set PyTorch seed for consistent RNG behavior - match original exactly
    if (comm->me == 0 && debug) {
      torch::manual_seed(666);
      if (torch::cuda::is_available() && use_gpu) {
        torch::cuda::manual_seed_all(666);
      }
      std::cout << "[PAIR_IANN_MULTI_GPU] Seed set to 666" << std::endl;
    }
    
    models[0]->eval();
    
    // Use same tuple format as original pair_iann.cpp
    auto output = models[0]->forward({
        num_atoms_tensor.to(torch::kInt64),
        atomic_numbers_tensor.to(torch::kInt64),
        positions_tensor.to(torch::kFloat32),
        cell_tensor.to(torch::kFloat32),
        edge_indices_tensor.to(torch::kInt64),
        edge_vectors_tensor.to(torch::kFloat32),
        num_edges_tensor.to(torch::kInt64)
    }).toGenericDict();
    
    // Extract energy and forces
    torch::Tensor energy_tensor = output.at("energy").toTensor();
    torch::Tensor forces_tensor = output.at("forces").toTensor();
    
    // Try to get atomic energies if available
    torch::Tensor atomic_energy_tensor;
    bool has_atomic_energy = false;
    if (output.contains("atomic_energy")) {
      atomic_energy_tensor = output.at("atomic_energy").toTensor();
      has_atomic_energy = true;
    }
    
    // Move tensors back to CPU if using GPU
    if (use_gpu) {
      energy_tensor = energy_tensor.to(torch::kCPU);
      forces_tensor = forces_tensor.to(torch::kCPU);
      if (has_atomic_energy) {
        atomic_energy_tensor = atomic_energy_tensor.to(torch::kCPU);
      }
    }
    
    // Handle forces tensor - expect compact_size atoms, scatter back to original indices
    auto forces_accessor = forces_tensor.accessor<float, 2>();
    if (forces_tensor.size(0) == compact_size) {
      // Scatter forces back to original atom indices, only for local atoms
      for (int i = 0; i < compact_size; i++) {
        int atom_idx = compact_atoms[i];
        if (atom_idx < nlocal) {  // Only write forces for local atoms
          f[atom_idx][0] = forces_accessor[i][0];
          f[atom_idx][1] = forces_accessor[i][1];
          f[atom_idx][2] = forces_accessor[i][2];
        }
      }
    } else {
      std::ostringstream msg;
      msg << "[PAIR_IANN_MULTI_GPU] Model returned forces of shape [" << forces_tensor.size(0)
          << ", " << forces_tensor.size(1) << "], expected " << compact_size;
      error->all(FLERR, msg.str());
    }
    
    // Set energy in LAMMPS if requested
    if (eflag_global) {
      if (debug) {
        std::cout << "[PAIR_IANN_MULTI_GPU] Rank " << comm->me << " GPU " << current_gpu_id << " - Energy tensor dim: " << energy_tensor.dim() 
                  << ", size: [" << energy_tensor.size(0) << "], nlocal: " << nlocal 
                  << ", nall: " << nall << ", has_atomic_energy: " << has_atomic_energy << std::endl;
      }
      
      if (has_atomic_energy) {
        // Use atomic_energy if available - sum only local atoms from compact subgraph
        eng_vdwl = 0.0;
        for (int i = 0; i < compact_size; i++) {
          int atom_idx = compact_atoms[i];
          if (atom_idx < nlocal) {  // Only sum energy for local atoms
            eng_vdwl += atomic_energy_tensor[i].item<double>();
          }
        }
        if (debug) {
          std::cout << "[PAIR_IANN_MULTI_GPU] Rank " << comm->me << " GPU " << current_gpu_id << " - Atomic energy sum (nlocal): " << eng_vdwl << std::endl;
        }
      } else {
        // Model returned scalar energy - use directly
        eng_vdwl = energy_tensor.item<double>();
        if (debug) {
          std::cout << "[PAIR_IANN_MULTI_GPU] Rank " << comm->me << " GPU " << current_gpu_id << " - Scalar energy: " << eng_vdwl << std::endl;
        }
      }
    }
    
  } catch (const c10::Error& e) {
    error->all(FLERR, "[PAIR_IANN_MULTI_GPU] Error running ML model: " + std::string(e.what()));
  }
  
  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairIANNMultiGPU::allocate() 
{
  allocated = 1;
  int n = atom->ntypes;
  
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;
  
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
}

/* ---------------------------------------------------------------------- */

void PairIANNMultiGPU::settings(int narg, char **arg) 
{
  if (narg < 3) error->all(FLERR, "[PAIR_IANN_MULTI_GPU] Illegal pair_style command: need model type and model path");
  
  int iarg = 0;
  
  // Get model type
  model_type = utils::strdup(arg[iarg++]);
  
  // Get model path
  model_path = utils::strdup(arg[iarg++]);
  
  // Get cutoff
  cutoff = utils::numeric(FLERR, arg[iarg++], false, lmp);
  
  // Check for debug flag
  for (int i = iarg; i < narg; i++) {
    if (strcmp(arg[i], "debug") == 0) {
      debug = true;
      if (comm->me == 0) {
        std::cout << "[PAIR_IANN_MULTI_GPU] Rank " << comm->me << " GPU " << current_gpu_id << " - Debug mode enabled" << std::endl;
      }
    }
  }
  
  // Load the model on all available GPUs
  try {
    if (use_gpu && num_gpus > 0) {
      models.resize(num_gpus);
      for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        // Load model on CPU first
        models[gpu_id] = std::make_shared<torch::jit::Module>(torch::jit::load(model_path));
        models[gpu_id]->eval();
        
        // Move model to specific GPU
        models[gpu_id]->to(devices[gpu_id]);
        
        if (comm->me == 0) {
          std::cout << "[PAIR_IANN_MULTI_GPU] Rank " << comm->me << " GPU " << gpu_id << " - Model loaded successfully" << std::endl;
        }
      }
    } else {
      // CPU mode - load single model
      models.resize(1);
      models[0] = std::make_shared<torch::jit::Module>(torch::jit::load(model_path));
      models[0]->eval();
      
      if (comm->me == 0) {
        error->message(FLERR, "[PAIR_IANN_MULTI_GPU] Model loaded on CPU");
      }
    }
  }
  catch (const c10::Error& e) {
    error->all(FLERR, "[PAIR_IANN_MULTI_GPU] Error loading ML model: " + std::string(e.what()));
  }
}

/* ---------------------------------------------------------------------- */

void PairIANNMultiGPU::coeff(int narg, char **arg) 
{
  if (!allocated) allocate();
  
  // Read atom types
  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);
  
  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      setflag[i][j] = 1;
      count++;
    }
  }
  
  if (count == 0) error->all(FLERR, "[PAIR_IANN_MULTI_GPU] Incorrect args for pair coefficients");
}

/* ---------------------------------------------------------------------- */

void PairIANNMultiGPU::init_style() 
{
  // Request neighbor list for this pair style
  int irequest = neighbor->request(this, instance_me);

  // Enable a full neighbor list and include ghost atoms
  neighbor->requests[irequest]->enable_full();
  neighbor->requests[irequest]->enable_ghost();

  // Store the number of atom types for later use
  num_types = atom->ntypes;
}

/* ---------------------------------------------------------------------- */

double PairIANNMultiGPU::init_one(int i, int j) 
{
  if (setflag[i][j] == 0) {
    error->all(FLERR, "[PAIR_IANN_MULTI_GPU] All pair coeffs must be set for ML potentials");
  }
  
  return cutoff;
}

/* ---------------------------------------------------------------------- */

void PairIANNMultiGPU::build_edges(int inum, int *ilist, int *numneigh, int **firstneigh)
{
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;

  std::vector<std::array<int64_t, 2>> edge_indices;
  std::vector<std::array<double, 3>> edge_vectors;

  double cutoff_sq = cutoff * cutoff;
  
  // Include all atoms (nall) to consider all possible interactions
  std::set<int> unique_atoms;
  for (int i = 0; i < nall; i++) {
    unique_atoms.insert(i);
  }
  
  // Create compact mapping: old_index -> compact_index
  compact_map.clear();
  compact_atoms.clear();
  int compact_idx = 0;
  for (int atom_idx : unique_atoms) {
    compact_map[atom_idx] = compact_idx;
    compact_atoms.push_back(atom_idx);
    compact_idx++;
  }
  
  // Build edges for all atoms (nall) - consider all possible interactions
  for (int i = 0; i < nall; i++) {
    for (int jj = 0; jj < numneigh[i]; jj++) {
      int j = firstneigh[i][jj] & NEIGHMASK;

      double dx = atom->x[j][0] - atom->x[i][0];
      double dy = atom->x[j][1] - atom->x[i][1];
      double dz = atom->x[j][2] - atom->x[i][2];
      domain->minimum_image(dx, dy, dz);

      double rsq = dx*dx + dy*dy + dz*dz;
      if (rsq > cutoff_sq) continue;
      
      // Include all edges (both local-local and local-ghost)
      auto it_i = compact_map.find(i);
      auto it_j = compact_map.find(j);
      
      if (it_i != compact_map.end() && it_j != compact_map.end()) {
        edge_indices.push_back({it_i->second, it_j->second});
        edge_vectors.push_back({static_cast<float>(dx), static_cast<float>(dy), static_cast<float>(dz)});
      }
    }
  }

  // Now create tensors
  int num_edges = edge_indices.size();
  
  edge_indices_tensor = torch::empty({num_edges, 2}, torch::kInt64);
  edge_vectors_tensor = torch::empty({num_edges, 3}, torch::kFloat32);

  auto edge_indices_accessor = edge_indices_tensor.accessor<int64_t, 2>();
  auto edge_vectors_accessor = edge_vectors_tensor.accessor<float, 2>();

  for (int k = 0; k < num_edges; k++) {
    edge_indices_accessor[k][0] = edge_indices[k][0];
    edge_indices_accessor[k][1] = edge_indices[k][1];

    edge_vectors_accessor[k][0] = edge_vectors[k][0];
    edge_vectors_accessor[k][1] = edge_vectors[k][1];
    edge_vectors_accessor[k][2] = edge_vectors[k][2];
  }

  // Scalar tensor for number of edges
  num_edges_tensor = torch::tensor((int64_t)num_edges, torch::kInt64);
}

/* ---------------------------------------------------------------------- */

void PairIANNMultiGPU::distribute_atoms_to_gpus()
{
  int nlocal = atom->nlocal;
  
  // Initialize atom groups for each GPU
  atom_groups.resize(num_gpus);
  for (int i = 0; i < num_gpus; i++) {
    atom_groups[i].clear();
  }
  
  // Simple round-robin distribution
  for (int i = 0; i < nlocal; i++) {
    int gpu_id = i % num_gpus;
    atom_groups[gpu_id].push_back(i);
  }
  
}

/* ---------------------------------------------------------------------- */

void PairIANNMultiGPU::distribute_edges_to_gpus()
{
  int num_edges = edge_indices_tensor.size(0);
  
  // Initialize edge groups for each GPU
  edge_groups.resize(num_gpus);
  for (int i = 0; i < num_gpus; i++) {
    edge_groups[i].clear();
  }
  
  // Distribute edges based on which atoms they connect
  auto edge_indices_accessor = edge_indices_tensor.accessor<int64_t, 2>();
  for (int edge_id = 0; edge_id < num_edges; edge_id++) {
    int atom_i = edge_indices_accessor[edge_id][0];
    int atom_j = edge_indices_accessor[edge_id][1];
    
    // Find which GPU has atom_i
    int gpu_id = -1;
    for (int gid = 0; gid < num_gpus; gid++) {
      if (std::find(atom_groups[gid].begin(), atom_groups[gid].end(), atom_i) != atom_groups[gid].end()) {
        gpu_id = gid;
        break;
      }
    }
    
    if (gpu_id >= 0) {
      edge_groups[gpu_id].push_back(edge_id);
    }
  }
  
}

/* ---------------------------------------------------------------------- */

void PairIANNMultiGPU::run_inference_on_gpu(int gpu_id, int atom_start, int atom_end, int edge_start, int edge_end)
{
  try {
    // Set the current CUDA device for this thread
    if (use_gpu && torch::cuda::is_available()) {
      // Set CUDA device using runtime API
#ifdef __CUDACC__
      cudaSetDevice(gpu_id);
#endif
    } else if (gpu_id != 0) {
      // Fallback: if no GPU available, only allow gpu_id = 0
      if (comm->me == 0) {
        error->warning(FLERR, "[PAIR_IANN_MULTI_GPU] CUDA not available, falling back to CPU for GPU > 0");
      }
      return;
    }
    
    int nlocal = atom->nlocal;
    double **x = atom->x;
    int *type = atom->type;
    
    // Get atoms assigned to this GPU
    const auto& gpu_atoms = atom_groups[gpu_id];
    int num_atoms_gpu = gpu_atoms.size();
    
    if (num_atoms_gpu == 0) {
      // No atoms assigned to this GPU, return zero results
      energy_promises[gpu_id].set_value(torch::tensor({0.0}, torch::kFloat32));
      forces_promises[gpu_id].set_value(torch::zeros({nlocal, 3}, torch::kFloat32));
      return;
    }
    
    // Convert atom types to atomic numbers for this GPU's atoms
    std::vector<int> type_to_Z(atom->ntypes + 1); 
    for (int itype = 1; itype <= atom->ntypes; ++itype) {
        double mass = atom->mass[itype];
        type_to_Z[itype] = get_atomic_number_global(mass);
    }
    
    std::vector<int64_t> atomic_numbers_gpu(num_atoms_gpu);
    for (int i = 0; i < num_atoms_gpu; i++) {
        int atom_idx = gpu_atoms[i];
        atomic_numbers_gpu[i] = type_to_Z[atom->type[atom_idx]];
    }
    torch::Tensor atomic_numbers_tensor = torch::from_blob(atomic_numbers_gpu.data(), {num_atoms_gpu}, torch::TensorOptions().dtype(torch::kInt64)).clone();

    // Convert atom positions to tensor for this GPU's atoms
    torch::Tensor positions_tensor = torch::zeros({num_atoms_gpu, 3}, torch::kFloat32);
    auto coord_acc = positions_tensor.accessor<float, 2>();
    for (int i = 0; i < num_atoms_gpu; i++) {
        int atom_idx = gpu_atoms[i];
        coord_acc[i][0] = static_cast<float>(x[atom_idx][0]);
        coord_acc[i][1] = static_cast<float>(x[atom_idx][1]);
        coord_acc[i][2] = static_cast<float>(x[atom_idx][2]);
    }

    // Convert box to tensor (3,3) - same for all GPUs
    torch::Tensor cell_tensor = torch::zeros({3,3}, torch::kFloat32);
    auto cell_acc = cell_tensor.accessor<float,2>();
    
    double *boxlo = domain->boxlo;
    double *boxhi = domain->boxhi;
    double xy = domain->xy;
    double xz = domain->xz;
    double yz = domain->yz;
    
    cell_acc[0][0] = static_cast<float>(boxhi[0] - boxlo[0]);
    cell_acc[0][1] = 0.0f;
    cell_acc[0][2] = 0.0f;
    cell_acc[1][0] = static_cast<float>(xy);
    cell_acc[1][1] = static_cast<float>(boxhi[1] - boxlo[1]);
    cell_acc[1][2] = 0.0f;
    cell_acc[2][0] = static_cast<float>(xz);
    cell_acc[2][1] = static_cast<float>(yz);
    cell_acc[2][2] = static_cast<float>(boxhi[2] - boxlo[2]);

    // Get edges assigned to this GPU and remap to local atom indices
    const auto& gpu_edges = edge_groups[gpu_id];

    // Build map from global atom index -> local per-GPU index
    std::unordered_map<int, int> global_to_local;
    global_to_local.reserve(num_atoms_gpu * 2);
    for (int i = 0; i < num_atoms_gpu; i++) {
      global_to_local[gpu_atoms[i]] = i;
    }

    // Access global edge tensors (constructed in build_edges)
    auto global_edge_idx = this->edge_indices_tensor.accessor<int64_t, 2>();
    auto global_edge_vec = this->edge_vectors_tensor.accessor<float, 2>();

    // Collect only edges whose endpoints both reside on this GPU, with remapped indices
    std::vector<std::array<int64_t, 2>> local_edges;
    std::vector<std::array<float, 3>> local_edge_vecs;
    local_edges.reserve(gpu_edges.size());
    local_edge_vecs.reserve(gpu_edges.size());

    for (int i = 0; i < (int)gpu_edges.size(); i++) {
      int edge_idx = gpu_edges[i];
      int gi = static_cast<int>(global_edge_idx[edge_idx][0]);
      int gj = static_cast<int>(global_edge_idx[edge_idx][1]);

      auto it_i = global_to_local.find(gi);
      auto it_j = global_to_local.find(gj);
      if (it_i == global_to_local.end() || it_j == global_to_local.end()) {
        // Skip cross-GPU edges for this local forward
        continue;
      }

      local_edges.push_back({(int64_t)it_i->second, (int64_t)it_j->second});
      local_edge_vecs.push_back({
        global_edge_vec[edge_idx][0],
        global_edge_vec[edge_idx][1],
        global_edge_vec[edge_idx][2]
      });
    }

    int num_edges_gpu = static_cast<int>(local_edges.size());

    // Create per-GPU edge tensors
    torch::Tensor edge_indices_tensor = torch::empty({num_edges_gpu, 2}, torch::kInt64);
    torch::Tensor edge_vectors_tensor = torch::empty({num_edges_gpu, 3}, torch::kFloat32);

    if (num_edges_gpu > 0) {
      auto edge_indices_accessor = edge_indices_tensor.accessor<int64_t, 2>();
      auto edge_vectors_accessor = edge_vectors_tensor.accessor<float, 2>();
      for (int k = 0; k < num_edges_gpu; k++) {
        edge_indices_accessor[k][0] = local_edges[k][0];
        edge_indices_accessor[k][1] = local_edges[k][1];
        edge_vectors_accessor[k][0] = local_edge_vecs[k][0];
        edge_vectors_accessor[k][1] = local_edge_vecs[k][1];
        edge_vectors_accessor[k][2] = local_edge_vecs[k][2];
      }
    }

    torch::Tensor num_edges_tensor = torch::tensor((int64_t)num_edges_gpu, torch::kInt64);
    torch::Tensor num_atoms_tensor = torch::tensor((int64_t)num_atoms_gpu, torch::kInt64);

    // Validate edge indices before moving to GPU
    if (num_edges_gpu > 0) {
      auto edge_flat = edge_indices_tensor.reshape({-1});
      auto max_idx = edge_flat.max().item<int64_t>();
      auto min_idx = edge_flat.min().item<int64_t>();
      
      if (min_idx < 0 || max_idx >= num_atoms_gpu) {
        // Return zero results for this GPU
        energy_promises[gpu_id].set_value(torch::tensor({0.0}, torch::kFloat32));
        forces_promises[gpu_id].set_value(torch::zeros({nlocal, 3}, torch::kFloat32));
        return;
      }
    }

    // Move tensors to this GPU
    atomic_numbers_tensor = atomic_numbers_tensor.to(devices[gpu_id]);
    positions_tensor = positions_tensor.to(devices[gpu_id]);
    cell_tensor = cell_tensor.to(devices[gpu_id]);
    edge_indices_tensor = edge_indices_tensor.to(devices[gpu_id]);
    edge_vectors_tensor = edge_vectors_tensor.to(devices[gpu_id]);
    num_edges_tensor = num_edges_tensor.to(devices[gpu_id]);
    num_atoms_tensor = num_atoms_tensor.to(devices[gpu_id]);
    
    // Run inference
    models[gpu_id]->eval();
    
    auto output = models[gpu_id]->forward({
        num_atoms_tensor.to(torch::kInt64),
        atomic_numbers_tensor.to(torch::kInt64),
        positions_tensor.to(torch::kFloat32),
        cell_tensor.to(torch::kFloat32),
        edge_indices_tensor.to(torch::kInt64),
        edge_vectors_tensor.to(torch::kFloat32),
        num_edges_tensor.to(torch::kInt64)
    }).toGenericDict();
    
    // Extract results
    auto energy = output.at("energy").toTensor().cpu();
    auto forces = output.at("forces").toTensor().cpu();
    
    // Create full-size forces tensor for this GPU's contribution
    torch::Tensor full_forces = torch::zeros({nlocal, 3}, torch::kFloat32);
    auto full_forces_accessor = full_forces.accessor<float, 2>();
    auto forces_accessor = forces.accessor<float, 2>();
    
    for (int i = 0; i < num_atoms_gpu; i++) {
      int atom_idx = gpu_atoms[i];
      full_forces_accessor[atom_idx][0] = forces_accessor[i][0];
      full_forces_accessor[atom_idx][1] = forces_accessor[i][1];
      full_forces_accessor[atom_idx][2] = forces_accessor[i][2];
    }
    
    // Set promises
    energy_promises[gpu_id].set_value(energy);
    forces_promises[gpu_id].set_value(full_forces);
    
    // Handle variances if they exist
    if (output.find("energy_variance") != output.end()) {
      auto energy_var = output.at("energy_variance").toTensor().cpu();
      energy_var_promises[gpu_id].set_value(energy_var);
    }
    
    if (output.find("forces_variance") != output.end()) {
      auto forces_var = output.at("forces_variance").toTensor().cpu();
      torch::Tensor full_forces_var = torch::zeros({nlocal, 3}, torch::kFloat32);
      auto full_forces_var_accessor = full_forces_var.accessor<float, 2>();
      auto forces_var_accessor = forces_var.accessor<float, 2>();
      
      for (int i = 0; i < num_atoms_gpu; i++) {
        int atom_idx = gpu_atoms[i];
        full_forces_var_accessor[atom_idx][0] = forces_var_accessor[i][0];
        full_forces_var_accessor[atom_idx][1] = forces_var_accessor[i][1];
        full_forces_var_accessor[atom_idx][2] = forces_var_accessor[i][2];
      }
      forces_var_promises[gpu_id].set_value(full_forces_var);
    }
    
    if (output.find("atomic_energy_variance") != output.end()) {
      auto atomic_energy_var = output.at("atomic_energy_variance").toTensor().cpu();
      torch::Tensor full_atomic_energy_var = torch::zeros({nlocal}, torch::kFloat32);
      auto full_atomic_energy_var_accessor = full_atomic_energy_var.accessor<float, 1>();
      auto atomic_energy_var_accessor = atomic_energy_var.accessor<float, 1>();
      
      for (int i = 0; i < num_atoms_gpu; i++) {
        int atom_idx = gpu_atoms[i];
        full_atomic_energy_var_accessor[atom_idx] = atomic_energy_var_accessor[i];
      }
      atomic_energy_var_promises[gpu_id].set_value(full_atomic_energy_var);
    }
    
  } catch (const c10::Error& e) {
    if (comm->me == 0) {
      std::string msg = "[PAIR_IANN_MULTI_GPU] Error on GPU " + std::to_string(gpu_id) + ": ";
      msg += e.what();
      error->warning(FLERR, msg.c_str());
    }
    
    // Set zero results on error
    energy_promises[gpu_id].set_value(torch::tensor({0.0}, torch::kFloat32));
    forces_promises[gpu_id].set_value(torch::zeros({atom->nlocal, 3}, torch::kFloat32));
  }
}

/* ---------------------------------------------------------------------- */

void PairIANNMultiGPU::collect_results_from_gpus()
{
  int nlocal = atom->nlocal;
  double **f = atom->f;
  
  try {
    // Collect energy results from all GPUs
    double total_energy = 0.0;
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
      auto energy_future = energy_promises[gpu_id].get_future();
      auto energy = energy_future.get();
      total_energy += energy.item<double>();
    }
    
    // Collect forces results from all GPUs
    torch::Tensor total_forces = torch::zeros({nlocal, 3}, torch::kFloat32);
    auto total_forces_accessor = total_forces.accessor<float, 2>();
    
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
      auto forces_future = forces_promises[gpu_id].get_future();
      auto forces = forces_future.get();
      auto forces_accessor = forces.accessor<float, 2>();
      
      for (int i = 0; i < nlocal; i++) {
        total_forces_accessor[i][0] += forces_accessor[i][0];
        total_forces_accessor[i][1] += forces_accessor[i][1];
        total_forces_accessor[i][2] += forces_accessor[i][2];
      }
    }
    
    // Copy forces to LAMMPS force array
    for (int i = 0; i < nlocal; i++) {
      f[i][0] = total_forces_accessor[i][0];
      f[i][1] = total_forces_accessor[i][1];
      f[i][2] = total_forces_accessor[i][2];
    }
    
    // Set energy in LAMMPS if requested
    if (eflag_global) {
      eng_vdwl = total_energy;
    }
    
    // Handle ensemble statistics if available
    bool has_variances = false;
    double total_energy_variance = 0.0;
    torch::Tensor total_forces_variance = torch::zeros({nlocal, 3}, torch::kFloat32);
    torch::Tensor total_atomic_energy_variance = torch::zeros({nlocal}, torch::kFloat32);
    
    // Check if any GPU has variance information
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
      try {
        auto energy_var_future = energy_var_promises[gpu_id].get_future();
        auto energy_var = energy_var_future.get();
        total_energy_variance += energy_var.item<double>();
        has_variances = true;
      } catch (...) {
        // No variance data from this GPU
      }
    }
    
    if (has_variances) {
      // Collect forces variance
      auto total_forces_var_accessor = total_forces_variance.accessor<float, 2>();
      for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        try {
          auto forces_var_future = forces_var_promises[gpu_id].get_future();
          auto forces_var = forces_var_future.get();
          auto forces_var_accessor = forces_var.accessor<float, 2>();
          
          for (int i = 0; i < nlocal; i++) {
            total_forces_var_accessor[i][0] += forces_var_accessor[i][0];
            total_forces_var_accessor[i][1] += forces_var_accessor[i][1];
            total_forces_var_accessor[i][2] += forces_var_accessor[i][2];
          }
        } catch (...) {
          // No forces variance data from this GPU
        }
      }
      
      // Collect atomic energy variance
      auto total_atomic_energy_var_accessor = total_atomic_energy_variance.accessor<float, 1>();
      for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        try {
          auto atomic_energy_var_future = atomic_energy_var_promises[gpu_id].get_future();
          auto atomic_energy_var = atomic_energy_var_future.get();
          auto atomic_energy_var_accessor = atomic_energy_var.accessor<float, 1>();
          
          for (int i = 0; i < nlocal; i++) {
            total_atomic_energy_var_accessor[i] += atomic_energy_var_accessor[i];
          }
        } catch (...) {
          // No atomic energy variance data from this GPU
        }
      }
      
      // Calculate ensemble statistics
      energy_variance = total_energy_variance;
      
      // Calculate variance of force magnitudes
      auto force_norms = torch::norm(total_forces, 2, 1);
      force_variance = torch::var(force_norms, 0).item<double>();
      
      max_energy_variance = total_atomic_energy_variance.max().item<double>();
      
      // Calculate maximum force variance
      max_force_variance = 0.0;
      for (int i = 0; i < nlocal; i++) {
        double atom_max_var = std::max({total_forces_var_accessor[i][0], 
                                      total_forces_var_accessor[i][1], 
                                      total_forces_var_accessor[i][2]});
        max_force_variance = std::max(max_force_variance, atom_max_var);
      }
      
      // Update global properties for thermo output
      global_properties[0] = energy_variance;
      global_properties[1] = force_variance;
      global_properties[2] = max_energy_variance;
      global_properties[3] = max_force_variance;
      
    }
    
    // Reset promises for next iteration
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
      energy_promises[gpu_id] = std::promise<torch::Tensor>();
      forces_promises[gpu_id] = std::promise<torch::Tensor>();
      energy_var_promises[gpu_id] = std::promise<torch::Tensor>();
      forces_var_promises[gpu_id] = std::promise<torch::Tensor>();
      atomic_energy_var_promises[gpu_id] = std::promise<torch::Tensor>();
    }
    
  } catch (const std::exception& e) {
    error->all(FLERR, "[PAIR_IANN_MULTI_GPU] Error collecting results from GPUs: " + std::string(e.what()));
  }
}