/* ----------------------------------------------------------------------
   Inherits from LAMMPS Pair class
   Uses LibTorch C++ API to interface with trained models
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

#include "pair_iann.h"

#include <cmath>
#include <cstring>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <unistd.h>

// LibTorch headers
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>
#include <vector>

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

std::map<int, int> type_to_Z;
std::map<double, int> mass_to_Z = {
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

int get_atomic_number(double mass) {
    for (const auto& [m, z] : mass_to_Z) {
        if (fabs(mass - m) < 0.1) {  // 0.1 u tolerance
            return z;
        }
    }
    return -1;  // Unknown
}


/* ---------------------------------------------------------------------- */

PairIANN::PairIANN(LAMMPS *lmp) : Pair(lmp),
  use_gpu(false),  // Initialize to false by default
  device(torch::Device(torch::kCPU))  // Initialize to CPU by default
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

  // Check CUDA environment
  const char* cuda_path = getenv("CUDA_HOME");
  const char* ld_path = getenv("LD_LIBRARY_PATH");
  
  if (comm->me == 0) {
    if (!cuda_path) {
      error->warning(FLERR, "[PAIR_IANN] CUDA_HOME environment variable not set");
    } else {
      // Check CUDA version
      std::string cuda_version;
      std::ifstream version_file(std::string(cuda_path) + "/version.txt");
      if (version_file.is_open()) {
        std::getline(version_file, cuda_version);
        error->message(FLERR, ("[PAIR_IANN] Found CUDA version: " + cuda_version).c_str());
      }
    }
    if (!ld_path) {
      error->warning(FLERR, "[PAIR_IANN] LD_LIBRARY_PATH environment variable not set");
    }
  }

  // Try to initialize CUDA with a small test case
  try {
    if (torch::cuda::is_available()) {
      // Create a small test case with 20 atoms
      int ntest = 20;
      torch::Tensor test_positions = torch::zeros({ntest, 3}, torch::kFloat32);
      
      // Try to move tensors to GPU
      test_positions = test_positions.cuda();
      
      // Perform a simple operation on GPU
      test_positions = test_positions + 1.0;
      
      // Move back to CPU and verify
      test_positions = test_positions.cpu();
      if (test_positions[0][0].item<float>() == 1.0) {
        use_gpu = true;
        device = torch::Device(torch::kCUDA);
        if (comm->me == 0) {
          error->message(FLERR, "[PAIR_IANN] GPU available");
        }
      }
    }
  } catch (const c10::Error& e) {
    use_gpu = false;
    device = torch::Device(torch::kCPU);
    if (comm->me == 0) {
      std::string msg = "[PAIR_IANN] GPU test failed: ";
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
    error->message(FLERR, "[PAIR_IANN] Using CPU mode");
  }
}

/* ---------------------------------------------------------------------- */

PairIANN::~PairIANN() 
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
  
  if (model_type) delete[] model_type;
  if (model_path) delete[] model_path;
  
  // Clean up global properties
  delete[] global_properties;
  
  // LibTorch cleanup is automatic via shared_ptr
}

/* ---------------------------------------------------------------------- */

void PairIANN::compute(int eflag, int vflag) 
{
  ev_init(eflag, vflag);
  
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int natoms_global = atom->natoms;  // Total number of atoms in the system

  static bool cutoff_printed = false;  // Static variable to track if cutoff has been printed
  if (comm->me == 0 && !cutoff_printed) {
    std::cout << "[PAIR_IANN] Using cutoff: " << cutoff << " Angstrom" << std::endl;
    cutoff_printed = true;  // Mark that we've printed the cutoff
  }
  
  // Convert number of atoms to tensor
  torch::Tensor num_atoms_tensor = torch::tensor({nlocal}, torch::kInt64);

  // Convert atom types to atomic numbers
  std::vector<int> type_to_Z(atom->ntypes + 1); 
  for (int itype = 1; itype <= atom->ntypes; ++itype) {
      double mass = atom->mass[itype];
      type_to_Z[itype] = get_atomic_number(mass);
  }
  std::vector<int64_t> atomic_numbers(nlocal);
  for (int i = 0; i < nlocal; ++i) {
      atomic_numbers[i] = type_to_Z[atom->type[i]];
  }
  torch::Tensor atomic_numbers_tensor = torch::from_blob(atomic_numbers.data(), {nlocal}, torch::TensorOptions().dtype(torch::kInt64)).clone();

  // Convert atom positions to tensor (nlocal, 3)
  torch::Tensor positions_tensor = torch::zeros({nlocal, 3}, torch::kFloat32);
  auto coord_acc = positions_tensor.accessor<float, 2>();
  for (int i = 0; i < nlocal; i++) {
      coord_acc[i][0] = static_cast<float>(x[i][0]);
      coord_acc[i][1] = static_cast<float>(x[i][1]);
      coord_acc[i][2] = static_cast<float>(x[i][2]);
  }

  // Convert box to tensor (3,3) for both orthogonal and non-orthogonal boxes
  torch::Tensor cell_tensor = torch::zeros({3,3}, torch::kFloat32);
  auto cell_acc = cell_tensor.accessor<float,2>();
  
  // Get box vectors
  double *boxlo = domain->boxlo;
  double *boxhi = domain->boxhi;
  double xy = domain->xy;
  double xz = domain->xz;
  double yz = domain->yz;
  
  // Set cell matrix elements
  cell_acc[0][0] = static_cast<float>(boxhi[0] - boxlo[0]);  // a_x
  cell_acc[0][1] = 0.0f;
  cell_acc[0][2] = 0.0f;

  cell_acc[1][0] = static_cast<float>(xy);                  // b_x
  cell_acc[1][1] = static_cast<float>(boxhi[1] - boxlo[1]); // b_y
  cell_acc[1][2] = 0.0f;

  cell_acc[2][0] = static_cast<float>(xz);                  // c_x
  cell_acc[2][1] = static_cast<float>(yz);                  // c_y
  cell_acc[2][2] = static_cast<float>(boxhi[2] - boxlo[2]); // c_z

  // Build edges based on neighborlist
  build_edges(list->inum, list->ilist, list->numneigh, list->firstneigh);

  // Move tensors to GPU if available
  if (use_gpu) {
    num_atoms_tensor = num_atoms_tensor.to(device);
    atomic_numbers_tensor = atomic_numbers_tensor.to(device);
    positions_tensor = positions_tensor.to(device);
    cell_tensor = cell_tensor.to(device);
    edge_indices_tensor = edge_indices_tensor.to(device);
    edge_vectors_tensor = edge_vectors_tensor.to(device);
    num_edges_tensor = num_edges_tensor.to(device);
  }

  // Debug: print input tensor sizes for sanity
  if (debug && comm->me == 0) {
    std::cout << "[PAIR_IANN] total_atoms=" << num_atoms_tensor << std::endl;
    std::cout << "[PAIR_IANN] atom_types=" << atomic_numbers_tensor << std::endl;
    std::cout << "[PAIR_IANN] atom_positions=" << positions_tensor << std::endl;
    std::cout << "[PAIR_IANN] cell=" << cell_tensor << std::endl;
    std::cout << "[PAIR_IANN] edge_indices=" << edge_indices_tensor << std::endl;
    std::cout << "[PAIR_IANN] edge_vectors=" << edge_vectors_tensor << std::endl;
    std::cout << "[PAIR_IANN] num_edges=" << num_edges_tensor << std::endl;
  }
  
  // Inference: call the scripted model with raw tensor inputs matching wrapper signature
  try {
    // Set PyTorch seed for consistent RNG behavior between PyTorch and LAMMPS
    // This ensures deterministic behavior when using random operations in the model

    if (comm->me == 0 && debug) {
      torch::manual_seed(666);
      if (torch::cuda::is_available() && use_gpu) {
        torch::cuda::manual_seed_all(666);
      }
      std::cout << "[PAIR_IANN] Seed set to 666" << std::endl;
    }
    
    // Ensure model is in evaluation mode for deterministic inference
    // This disables dropout operations and matches ASE interface behavior
    model->eval();
    
    auto output = model->forward({
        num_atoms_tensor.to(torch::kInt64),
        atomic_numbers_tensor.to(torch::kInt64),
        positions_tensor.to(torch::kFloat32),
        cell_tensor.to(torch::kFloat32),
        edge_indices_tensor.to(torch::kInt64),
        edge_vectors_tensor.to(torch::kFloat32),
        num_edges_tensor.to(torch::kInt64)
    }).toGenericDict();
    
    // Extract energy and forces
    auto energy = output.at("energy").toTensor();
    auto forces = output.at("forces").toTensor();
    
    // Move tensors back to CPU if using GPU
    if (use_gpu) {
      energy = energy.to(torch::kCPU);
      forces = forces.to(torch::kCPU);
    }
    
    // Extract variances if they exist (ensemble model)
    torch::Tensor energy_var, forces_var, atomic_energy_var;
    bool has_variances = false;
    if (output.find("energy_variance") != output.end() && 
        output.find("forces_variance") != output.end() &&
        output.find("atomic_energy_variance") != output.end()) {
        has_variances = true;
        energy_var = output.at("energy_variance").toTensor();
        forces_var = output.at("forces_variance").toTensor();
        atomic_energy_var = output.at("atomic_energy_variance").toTensor();
        
        // Move variance tensors back to CPU if using GPU
        if (use_gpu) {
          energy_var = energy_var.to(torch::kCPU);
          forces_var = forces_var.to(torch::kCPU);
          atomic_energy_var = atomic_energy_var.to(torch::kCPU);
        }
    }
    
    // Debug: print energy value and forces tensor shape
    if (debug && comm->me == 0) {
        double e_val = energy.item<double>();
        auto f_sizes = forces.sizes();
        std::cout << "[PAIR_IANN_DEBUG] returned energy = " << e_val << std::endl;
        std::cout << "[PAIR_IANN_DEBUG] forces shape = [" << f_sizes[0] << ", " << f_sizes[1] << "]" << std::endl;
        if (has_variances) {
            std::cout << "[PAIR_IANN_DEBUG] energy variance = " << energy_var.item<double>() << std::endl;
            std::cout << "[PAIR_IANN_DEBUG] forces variance shape = [" << forces_var.size(0) << ", " << forces_var.size(1) << "]" << std::endl;
            std::cout << "[PAIR_IANN_DEBUG] atomic energy variance shape = [" << atomic_energy_var.size(0) << "]" << std::endl;
        }
    }
    
    // Handle forces tensor
    auto forces_accessor = forces.accessor<float, 2>();
    if (forces.size(0) == nlocal) {
      // Model returned per-atom forces in local order
      for (int i = 0; i < nlocal; i++) {
        f[i][0] = forces_accessor[i][0];
        f[i][1] = forces_accessor[i][1];
        f[i][2] = forces_accessor[i][2];
      }
    } else {
      std::ostringstream msg;
      msg << "[PAIR_IANN] Model returned forces of shape [" << forces.size(0)
          << ", " << forces.size(1) << "], expected " << nlocal
          << " or " << num_edges_tensor.item<int64_t>();
      error->all(FLERR, msg.str());
    }
    
    // Set energy in LAMMPS if requested
    if (eflag_global) {
        eng_vdwl = energy.item<double>();
        if (has_variances) {
            energy_variance = energy_var.item<double>();
            
            // Calculate variance of force magnitudes
            auto force_norms = torch::norm(forces_var, 2, 1);  // Calculate L2 norm along dimension 1 (xyz components)
            force_variance = torch::var(force_norms, 0).item<double>(); // variance of force magnitudes
            
            max_energy_variance = atomic_energy_var.max().item<double>();
            
            // Calculate maximum force variance
            max_force_variance = 0.0;
            auto forces_var_accessor = forces_var.accessor<float, 2>();
            for (int i = 0; i < nlocal; i++) {
              double atom_max_var = std::max({forces_var_accessor[i][0], 
                                            forces_var_accessor[i][1], 
                                            forces_var_accessor[i][2]});
              max_force_variance = std::max(max_force_variance, atom_max_var);
            }
            
            // Update global properties for thermo output
            global_properties[0] = energy_variance;
            global_properties[1] = force_variance;
            global_properties[2] = max_energy_variance;
            global_properties[3] = max_force_variance;
            
            // Print variance statistics
            if (debug && comm->me == 0) {
              std::cout << "[PAIR_IANN_DEBUG] Energy variance: " << energy_variance << std::endl;
              std::cout << "[PAIR_IANN_DEBUG] Force variance: " << force_variance << std::endl;
              std::cout << "[PAIR_IANN_DEBUG] Maximum energy variance: " << max_energy_variance << std::endl;
              std::cout << "[PAIR_IANN_DEBUG] Maximum force variance: " << max_force_variance << std::endl;
            }
        }
    }
    
    // Per-atom energies and variances (if model provides them)
    if (eflag_atom) {
        if (output.find("atomic_energy") != output.end()) {
            auto atom_energies = output.at("atomic_energy").toTensor();
            if (use_gpu) {
              atom_energies = atom_energies.to(torch::kCPU);
            }
            auto energies_accessor = atom_energies.accessor<float, 1>();
            for (int i = 0; i < nlocal; i++) {
                eatom[i] = energies_accessor[i];
            }
        }
    }
    
    if (vflag_fdotr) virial_fdotr_compute();

    if (atomic_numbers_tensor.size(0) != nlocal || positions_tensor.size(0) != nlocal) {
      error->all(FLERR, "[PAIR_IANN] Tensor shape mismatch with nlocal atoms");
    }
    if (torch::any(torch::isnan(forces)).item<bool>()) {
      error->all(FLERR, "[PAIR_IANN] NaNs in predicted forces.");
    }
    if (forces.abs().max().item<float>() > 1e4) {
      error->all(FLERR, "[PAIR_IANN] Unphysically large force detected.");
    }
  }
  catch (const c10::Error& e) {
    error->all(FLERR, "[PAIR_IANN] Error running ML model: " + std::string(e.what()));
  }
}

/* ---------------------------------------------------------------------- */

void PairIANN::allocate() 
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

void PairIANN::settings(int narg, char **arg) 
{
  if (narg < 3) error->all(FLERR, "[PAIR_IANN] Illegal pair_style command: need model type and model path");
  
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
        error->message(FLERR, "[PAIR_IANN] Debug mode enabled");
    }
  }
  
  // Load the model
  try {
    // First try loading on CPU
    model = std::make_shared<torch::jit::Module>(torch::jit::load(model_path));
    model->eval();  // Set to inference mode
    
    // Move model to GPU if available
    if (use_gpu) {
      try {
        // Verify CUDA is still available
        if (!torch::cuda::is_available()) {
          throw std::runtime_error("[PAIR_IANN] CUDA is no longer available");
        }

        // Move model to GPU
        model->to(device);
        
        // Test the model on GPU with a small system
        if (comm->me == 0) {
          // Create test data for 20 atoms
          int ntest = 20;
          torch::Tensor test_positions = torch::zeros({ntest, 3}, torch::kFloat32);
          
          // Use the most common element from the actual atoms, or default to Carbon
          int test_element = 6; // Default to Carbon
          if (atom->ntypes > 0) {
            // Get the most common element from the system
            std::map<int, int> element_counts;
            for (int i = 1; i <= atom->ntypes; ++i) {
              double mass = atom->mass[i];
              int atomic_number = get_atomic_number(mass);
              if (atomic_number > 0) {
                element_counts[atomic_number]++;
              }
            }
            if (!element_counts.empty()) {
              // Use the most common element
              test_element = element_counts.begin()->first;
            }
          }
          torch::Tensor test_types = torch::ones({ntest}, torch::kInt64) * test_element;
          torch::Tensor test_cell = torch::eye(3, torch::kFloat32) * 10.0; // 10Å cubic cell
          
          // Create test edges (neighbor list)
          std::vector<std::array<int64_t, 2>> test_edges;
          for (int i = 0; i < ntest; i++) {
            for (int j = i + 1; j < ntest; j++) {
              test_edges.push_back({i, j});
            }
          }
          torch::Tensor test_edge_indices = torch::zeros({(int64_t)test_edges.size(), 2}, torch::kInt64);
          torch::Tensor test_edge_vectors = torch::zeros({(int64_t)test_edges.size(), 3}, torch::kFloat32);
          
          // Fill edge data
          for (size_t i = 0; i < test_edges.size(); i++) {
            test_edge_indices[i][0] = test_edges[i][0];
            test_edge_indices[i][1] = test_edges[i][1];
            test_edge_vectors[i][0] = 1.0; // Example edge vector
            test_edge_vectors[i][1] = 0.0;
            test_edge_vectors[i][2] = 0.0;
          }
          
          torch::Tensor test_num_edges = torch::tensor((int64_t)test_edges.size(), torch::kInt64);
          
          // Print test tensor shapes
          if (debug && comm->me == 0) {
            std::cout << "[PAIR_IANN_DEBUG] Test tensor shapes:" << std::endl;
            std::cout << "[PAIR_IANN_DEBUG] Positions: " << test_positions.sizes() << std::endl;
            std::cout << "[PAIR_IANN_DEBUG] Types: " << test_types.sizes() << std::endl;
            std::cout << "[PAIR_IANN_DEBUG] Cell: " << test_cell.sizes() << std::endl;
            std::cout << "[PAIR_IANN_DEBUG] Edge indices: " << test_edge_indices.sizes() << std::endl;
            std::cout << "[PAIR_IANN_DEBUG] Edge vectors: " << test_edge_vectors.sizes() << std::endl;
            std::cout << "[PAIR_IANN_DEBUG] Number of edges: " << test_num_edges.item<int64_t>() << std::endl;
          }
          
          // Move test data to GPU
          test_positions = test_positions.to(device);
          test_types = test_types.to(device);
          test_cell = test_cell.to(device);
          test_edge_indices = test_edge_indices.to(device);
          test_edge_vectors = test_edge_vectors.to(device);
          test_num_edges = test_num_edges.to(device);
          
          // Try running the model
          auto output = model->forward({
              torch::tensor({ntest}, torch::kInt64).to(device),
              test_types,
              test_positions,
              test_cell,
              test_edge_indices,
              test_edge_vectors,
              test_num_edges
          }).toGenericDict();
          
          // Move results back to CPU and verify
          auto energy = output.at("energy").toTensor().cpu();
          auto forces = output.at("forces").toTensor().cpu();
          
          if (!torch::any(torch::isnan(energy)).item<bool>() && 
              !torch::any(torch::isnan(forces)).item<bool>()) {
            error->message(FLERR, "[PAIR_IANN] GPU model test successful");
          } else {
            throw std::runtime_error("[PAIR_IANN] Model produced NaN values");
          }
        }
        
        if (comm->me == 0) {
          error->message(FLERR, "[PAIR_IANN] Using GPU acceleration");
        }
      } catch (const c10::Error& e) {
        // If GPU model loading or test fails, fall back to CPU
        use_gpu = false;
        device = torch::Device(torch::kCPU);
        model->to(device);
        if (comm->me == 0) {
          std::string msg = "[PAIR_IANN] Failed to run model on GPU: ";
          msg += e.what();
          msg += "\n[PAIR_IANN] Falling back to CPU mode.";
          error->warning(FLERR, msg.c_str());
        }
      }
    }
  }
  catch (const c10::Error& e) {
    error->all(FLERR, "[PAIR_IANN] Error loading ML model: " + std::string(e.what()));
  }
}

/* ---------------------------------------------------------------------- */

void PairIANN::coeff(int narg, char **arg) 
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
  
  if (count == 0) error->all(FLERR, "[PAIR_IANN] Incorrect args for pair coefficients");
}

/* ---------------------------------------------------------------------- */

void PairIANN::init_style() 
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

double PairIANN::init_one(int i, int j) 
{
  if (setflag[i][j] == 0) {
    error->all(FLERR, "[PAIR_IANN] All pair coeffs must be set for ML potentials");
  }
  
  return cutoff;
}

/* ---------------------------------------------------------------------- */

void PairIANN::build_edges(int inum, int *ilist, int *numneigh, int **firstneigh)
{
  int nlocal = atom->nlocal;

  std::vector<std::array<int64_t, 2>> edge_indices;
  std::vector<std::array<double, 3>> edge_vectors;

  double cutoff_sq = cutoff * cutoff;
  
  atom->map_init();
  atom->map_set();
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    for (int jj = 0; jj < numneigh[i]; jj++) {
      int j = firstneigh[i][jj] & NEIGHMASK;
      // if (j >= nlocal) continue;

      double dx = atom->x[j][0] - atom->x[i][0];
      double dy = atom->x[j][1] - atom->x[i][1];
      double dz = atom->x[j][2] - atom->x[i][2];
      domain->minimum_image(dx, dy, dz);

      double rsq = dx*dx + dy*dy + dz*dz;
      if (rsq > cutoff_sq) continue;
      
      if (j >= nlocal) {
        // Debugging line to check the ghost atom's global ID
        // std::cerr << "Attempting to map ghost atom with global ID: " << j << std::endl;
        int atom_tag = atom->tag[j];
        int local_j = atom->map(atom_tag);
        // std::cerr << "Local j: " << local_j << ". " << std::endl;
        if (local_j == -1) {
          std::cerr << "Error: Failed to map ghost atom with global ID " << j << " to a local index." << std::endl;
          throw std::runtime_error("Error: Failed to map ghost atom to local index (invalid atom ID).");
          //continue;  // Skip this iteration
        }
        j = local_j;
      }
      edge_indices.push_back({i, j});
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
  num_edges_tensor = torch::tensor((int64_t)num_edges, torch::kInt64);
}