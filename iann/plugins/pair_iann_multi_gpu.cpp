/* ----------------------------------------------------------------------
   Multi-GPU version of IANN LAMMPS interface
   Inherits from LAMMPS Pair class
   Uses LibTorch C++ API to interface with trained models across multiple GPUs
   Handles neighbor list construction, energy/force calculations

Usage:
  pair_style iann/multi_gpu painn model.pt 5.5

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
    throw std::runtime_error("[PAIR_IANN_MULTI_GPU] Unknown atomic number for mass: " + std::to_string(mass));
}

/* ---------------------------------------------------------------------- */

PairIANNMultiGPU::PairIANNMultiGPU(LAMMPS *lmp) : Pair(lmp),
  use_gpu(false),  // Initialize to false by default
  use_multi_gpu(false),  // Initialize to false by default
  local_gpus(0)
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

  // Initialize promises for multi-GPU communication
  // Will be resized to actual number of GPUs after GPU detection
  energy_promises.resize(1);  // Start with 1, will resize later
  forces_promises.resize(1);

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
      this->local_gpus = torch::cuda::device_count();
      int local_gpus = this->local_gpus;  // Local reference for convenience, number of GPUs per node

      int global_gpus = comm->nprocs;
      
      // Each rank uses only one GPU (rank-based assignment)
      int assigned_gpu_id = comm->me % local_gpus;  // Assign GPU based on rank

      // All ranks report their GPU assignment
      int node_id = comm->me / local_gpus;  // Calculate node ID based on GPUs per node
      std::cout << "[PAIR_IANN_MULTI_GPU] Running on GPU " << assigned_gpu_id << " of Node " << node_id << ". Total number of GPUs is " << global_gpus << std::endl;
      
      // Initialize device list for this rank's assigned GPU only
      devices.push_back(torch::Device(torch::kCUDA, assigned_gpu_id));
      
      // Test the assigned GPU
      try {
        torch::Device test_device(torch::kCUDA, assigned_gpu_id);
        int ntest = 20;
        torch::Tensor test_positions = torch::zeros({ntest, 3}, torch::kFloat32);
        
        // Try to move tensors to specific GPU
        test_positions = test_positions.to(test_device);
        
        // Perform a simple operation on GPU
        test_positions = test_positions + 1.0;
        
        // Move back to CPU and verify
        test_positions = test_positions.cpu();
        if (test_positions[0][0].item<float>() == 1.0) {
          std::cout << "[PAIR_IANN_MULTI_GPU] Rank " << comm->me << " successfully assigned to GPU " << assigned_gpu_id << " of Node " << node_id << std::endl;
        }
      } catch (const c10::Error& e) {
        std::cout << "[PAIR_IANN_MULTI_GPU] Rank " << comm->me << " failed to assign to GPU " << assigned_gpu_id << " of Node " << node_id << " - test failed: " << e.what() << std::endl;
      }
      
      // GPU mode
      if (local_gpus > 0) {
        use_gpu = true;
        use_multi_gpu = (local_gpus > 1);
        
        // Resize promises to actual number of GPUs per node
        energy_promises.resize(local_gpus);
        forces_promises.resize(local_gpus);
        if (comm->me == 0) {
          if (use_multi_gpu) {
            error->message(FLERR, "[PAIR_IANN_MULTI_GPU] Multi-GPU mode enabled ");
          } else {
            error->message(FLERR, "[PAIR_IANN_MULTI_GPU] Single GPU mode enabled");
          }
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
    this->local_gpus = 0;
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

  // Build edges based on neighborlist, got num_edges_tensor, edge_indices_tensor and edge_vectors_tensor
  build_edges(list->inum, list->ilist, list->numneigh, list->firstneigh);

  // Use all atoms for num_atoms_tensor
  num_atoms_tensor = torch::tensor({nall}, torch::kInt64);

  // Convert atom types to atomic_numbers_tensor
  std::vector<int> type_to_Z(atom->ntypes + 1); 
  for (int itype = 1; itype <= atom->ntypes; ++itype) {
      double mass = atom->mass[itype];
      type_to_Z[itype] = get_atomic_number_global(mass);
  }
  std::vector<int64_t> atomic_numbers(nall);
  for (int i = 0; i < nall; ++i) {
      atomic_numbers[i] = type_to_Z[atom->type[i]];
  }
  atomic_numbers_tensor = torch::from_blob(atomic_numbers.data(), {nall}, torch::TensorOptions().dtype(torch::kInt64)).clone();

  // Convert atom positions to positions_tensor
  positions_tensor = torch::zeros({nall, 3}, torch::kFloat32);
  auto coord_acc = positions_tensor.accessor<float, 2>();
  for (int i = 0; i < nall; i++) {
      coord_acc[i][0] = static_cast<float>(x[i][0]);
      coord_acc[i][1] = static_cast<float>(x[i][1]);
      coord_acc[i][2] = static_cast<float>(x[i][2]);
  }

  // Convert box to cell_tensor (3,3)
  cell_tensor = torch::zeros({3,3}, torch::kFloat32);
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

  if (torch::cuda::is_available() && use_gpu) {
    torch::cuda::manual_seed_all(666);
  } else {
    torch::manual_seed(666);
  }

  // Run inference
  try {
      if (use_multi_gpu && local_gpus > 1) {
        // Multi-GPU inference - each rank uses its assigned GPU
      if (debug) {
        std::cout << "[PAIR_IANN_MULTI_GPU] Rank " << comm->me << " Using multi-GPU inference with " << local_gpus << " GPUs per node" << std::endl;
      }
      
      // Each rank runs inference on its assigned GPU
      int assigned_gpu_id = comm->me % local_gpus;
      int node_id = comm->me / local_gpus;
      
      // Reset promises for this iteration
      energy_promises[0] = std::promise<torch::Tensor>();
      forces_promises[0] = std::promise<torch::Tensor>();
      
      // Run inference on assigned GPU
      run_inference_on_gpu(assigned_gpu_id, node_id, 0, 0, 0);
      
      // Collect results from this rank's GPU
      collect_results_from_gpus();
      
    } else {
      // Single GPU inference
      if (debug && comm->me == 0) {
        static bool printed = true;
        if (printed) {
        std::cout << "[PAIR_IANN_MULTI_GPU] Using single GPU inference" << std::endl;
        printed = false;
        }
      }
      
      // Move tensors to GPU before forward pass
      if (use_gpu) {
        num_atoms_tensor = num_atoms_tensor.to(devices[0]);
        atomic_numbers_tensor = atomic_numbers_tensor.to(devices[0]);
        positions_tensor = positions_tensor.to(devices[0]);
        cell_tensor = cell_tensor.to(devices[0]);
        edge_indices_tensor = edge_indices_tensor.to(devices[0]);
        edge_vectors_tensor = edge_vectors_tensor.to(devices[0]);
        num_edges_tensor = num_edges_tensor.to(devices[0]);
      }

      models[0]->eval();
      
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(num_atoms_tensor.to(torch::kInt64));
      inputs.push_back(atomic_numbers_tensor.to(torch::kInt64));
      inputs.push_back(positions_tensor.to(torch::kFloat32));
      inputs.push_back(cell_tensor.to(torch::kFloat32));
      inputs.push_back(edge_indices_tensor.to(torch::kInt64));
      inputs.push_back(edge_vectors_tensor.to(torch::kFloat32));
      inputs.push_back(num_edges_tensor.to(torch::kInt64));
      
      auto output = models[0]->forward(inputs).toGenericDict();
      
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
      
      // Handle forces tensor - expect nall atoms, only write forces for local atoms
      auto forces_accessor = forces_tensor.accessor<float, 2>();
      if (forces_tensor.size(0) == nall) {
        // Only write forces for local atoms
        for (int i = 0; i < nlocal; i++) {
          f[i][0] = forces_accessor[i][0];
          f[i][1] = forces_accessor[i][1];
          f[i][2] = forces_accessor[i][2];
        }
      } else {
        std::ostringstream msg;
        msg << "[PAIR_IANN_MULTI_GPU] Model returned forces of shape [" << forces_tensor.size(0)
            << ", " << forces_tensor.size(1) << "], expected " << nall;
        error->all(FLERR, msg.str());
      }
      
      // Set energy in LAMMPS if requested
      if (eflag_global) {
        if (debug && comm->me == 0) {
          std::cout << "[PAIR_IANN_MULTI_GPU] Energy tensor dim: " << energy_tensor.dim() 
                    << ", size: [" << energy_tensor.size(0) << "], nlocal: " << nlocal 
                    << ", nall: " << nall << ", has_atomic_energy: " << has_atomic_energy
                    << ", atomic_energy_tensor: [" << atomic_energy_tensor << "]" << std::endl;
        }
        
        if (has_atomic_energy) {
          // Use atomic_energy if available - sum only local atoms
          eng_vdwl = 0.0;
          for (int i = 0; i < nlocal; i++) {
            eng_vdwl += atomic_energy_tensor[i].item<float>();
          }
          if (debug && comm->me == 0) {
            std::cout << "[PAIR_IANN_MULTI_GPU] Rank " << comm->me << " Atomic energy sum (nlocal): " << eng_vdwl << std::endl;
          }
        } else {
          // Model returned scalar energy - use directly
          eng_vdwl = energy_tensor.item<float>();
          if (debug && comm->me == 0) {
            std::cout << "[PAIR_IANN_MULTI_GPU] Scalar energy: " << eng_vdwl << std::endl;
          }
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
      if (comm->me == 0)
        error->message(FLERR, "[PAIR_IANN_MULTI_GPU] Debug mode enabled");
    }
  }
  
  // Load the model on all available GPUs
  try {
    if (debug) {
      std::cout << "[PAIR_IANN_MULTI_GPU] Rank " << comm->me << " Model loading check: use_gpu=" << use_gpu << ", local_gpus=" << local_gpus << std::endl;
    }

    if (use_gpu && local_gpus > 0) {
      // Each rank loads model on its assigned GPU only
      models.resize(1);  // Only one model per rank
      int assigned_gpu_id = comm->me % local_gpus;  // Get assigned GPU ID
      int node_id = comm->me / local_gpus;  // Calculate node ID
      
      // Load model on CPU first
      models[0] = std::make_shared<torch::jit::Module>(torch::jit::load(model_path));
      models[0]->eval();
      
      // Move model to assigned GPU
      models[0]->to(devices[0]);
      
      if (debug) {
        std::cout << "[PAIR_IANN_MULTI_GPU] Rank " << comm->me << " Model loaded on GPU " << assigned_gpu_id << " of Node " << node_id << std::endl;
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
  
  // Build edges directly using nall indices
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
      edge_indices.push_back({static_cast<int64_t>(i), static_cast<int64_t>(j)});
      edge_vectors.push_back({static_cast<float>(dx), static_cast<float>(dy), static_cast<float>(dz)});
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
  num_edges_tensor = torch::tensor({(int64_t)num_edges}, torch::kInt64);
}

/* ---------------------------------------------------------------------- */

void PairIANNMultiGPU::run_inference_on_gpu(int assigned_gpu_id, int node_id, int atom_start, int edge_start, int edge_end)
{
  try {

    
    // Set the current CUDA device for this thread
    if (use_gpu && torch::cuda::is_available()) {
      // Set CUDA device using runtime API
      #ifdef __CUDACC__
            cudaSetDevice(assigned_gpu_id);
      #endif
    } 
    
    int nlocal = atom->nlocal;
    int nall = nlocal + atom->nghost;
    double **x = atom->x;
    int *type = atom->type;
    
    // Each rank processes its own atoms
    if (debug) {
      std::cout << "[PAIR_IANN_MULTI_GPU] Rank " << comm->me << " processing " << nall << " atoms (nlocal=" << atom->nlocal 
                << ", nghost=" << atom->nghost << ") on GPU " << assigned_gpu_id << " of Node " << node_id << std::endl;
    }
    
    // Use the global tensors created in compute() - no duplication needed
    torch::Tensor atomic_numbers_tensor = this->atomic_numbers_tensor;
    torch::Tensor positions_tensor = this->positions_tensor;
    torch::Tensor cell_tensor = this->cell_tensor;

    // Use the global edge tensors directly (no mapping needed with nall approach)
    torch::Tensor num_atoms_tensor = this->num_atoms_tensor;
    torch::Tensor num_edges_tensor = torch::tensor({(int64_t)this->edge_indices_tensor.size(0)}, torch::kInt64);
    
    // Use global edge tensors directly
    torch::Tensor edge_indices_tensor = this->edge_indices_tensor;
    torch::Tensor edge_vectors_tensor = this->edge_vectors_tensor;

    // Validate edge indices before moving to GPU
    if (edge_indices_tensor.size(0) > 0) {
      auto edge_flat = edge_indices_tensor.reshape({-1});
      auto max_idx = edge_flat.max().item<int64_t>();
      auto min_idx = edge_flat.min().item<int64_t>();
      
      if (min_idx < 0 || max_idx >= nall) {
        // Return zero results for this GPU
        energy_promises[0].set_value(torch::tensor({0.0}, torch::kFloat32));
        forces_promises[0].set_value(torch::zeros({nlocal, 3}, torch::kFloat32));
        return;
      }
    }

    // Move tensors to this rank's assigned GPU (devices[0] contains the assigned GPU)
    atomic_numbers_tensor = atomic_numbers_tensor.to(devices[0]);
    positions_tensor = positions_tensor.to(devices[0]);
    cell_tensor = cell_tensor.to(devices[0]);
    edge_indices_tensor = edge_indices_tensor.to(devices[0]);
    edge_vectors_tensor = edge_vectors_tensor.to(devices[0]);
    num_edges_tensor = num_edges_tensor.to(devices[0]);
    num_atoms_tensor = num_atoms_tensor.to(devices[0]);
    
    // Run inference
    models[0]->eval();
    
    auto output = models[0]->forward({
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
    
    // Only write forces for local atoms (0 to nlocal-1)
    for (int i = 0; i < nlocal; i++) {
      full_forces_accessor[i][0] = forces_accessor[i][0];
      full_forces_accessor[i][1] = forces_accessor[i][1];
      full_forces_accessor[i][2] = forces_accessor[i][2];
    }
    
    // Calculate energy as sum of local atoms only
    double local_energy = 0.0;
    if (output.contains("atomic_energy")) {
      auto atomic_energy = output.at("atomic_energy").toTensor().cpu();
      auto atomic_energy_accessor = atomic_energy.accessor<float, 1>();
      for (int i = 0; i < nlocal; i++) {
        local_energy += atomic_energy_accessor[i];
      }
    } else {
      // Use scalar energy if atomic energy not available
      local_energy = energy.item<double>();
    }
    
    // Set promises
    energy_promises[0].set_value(torch::tensor({local_energy}, torch::kFloat32));
    forces_promises[0].set_value(full_forces);
    
  } catch (const c10::Error& e) {
    if (comm->me == 0) {
      std::string msg = "[PAIR_IANN_MULTI_GPU] Error on rank " + std::to_string(comm->me) + " (GPU " + std::to_string(assigned_gpu_id) + "): ";
      msg += e.what();
      error->warning(FLERR, msg.c_str());
    }
    
    // Set zero results on error
    energy_promises[0].set_value(torch::tensor({0.0}, torch::kFloat32));
    forces_promises[0].set_value(torch::zeros({atom->nlocal, 3}, torch::kFloat32));
  }
}

/* ---------------------------------------------------------------------- */

void PairIANNMultiGPU::collect_results_from_gpus()
{
  int nlocal = atom->nlocal;
  double **f = atom->f;
  
  try {
    // Each rank collects results from its single assigned GPU
    auto energy_future = energy_promises[0].get_future();
    auto energy = energy_future.get();
    double total_energy = energy.item<double>();
    
    // Collect forces results from this rank's GPU
    auto forces_future = forces_promises[0].get_future();
    auto forces = forces_future.get();
    auto forces_accessor = forces.accessor<float, 2>();
    
    // Copy forces to LAMMPS force array
    for (int i = 0; i < nlocal; i++) {
      f[i][0] = forces_accessor[i][0];
      f[i][1] = forces_accessor[i][1];
      f[i][2] = forces_accessor[i][2];
    }
    
    // Set energy in LAMMPS
    eng_vdwl = total_energy;
    
    if (debug) {
      std::cout << "[PAIR_IANN_MULTI_GPU] Rank " << comm->me << " Results collection completed" << " eng_vdwl: " << eng_vdwl << std::endl;
    }
    
    // Reset promises for next iteration
    energy_promises[0] = std::promise<torch::Tensor>();
    forces_promises[0] = std::promise<torch::Tensor>();
    
  } catch (const std::exception& e) {
    error->all(FLERR, "[PAIR_IANN_MULTI_GPU] Error collecting results from GPUs: " + std::string(e.what()));
  }
}