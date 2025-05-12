/* ----------------------------------------------------------------------
   Inherits from LAMMPS Pair class
   Uses LibTorch C++ API to interface with trained models
   Handles neighbor list construction, energy/force calculations
------------------------------------------------------------------------- */

#include "pair_iann.h"

#include <cmath>
#include <cstring>
#include <vector>
#include <memory>
#include <iostream>

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

PairIANN::PairIANN(LAMMPS *lmp) : Pair(lmp) 
{
  restartinfo = 0;
  manybody_flag = 1;
  
  model_type = nullptr;
  model_path = nullptr;
  cutoff = 0.0;
  
  // Default settings
  comm_forward = 1;  // We need to communicate forces
  comm_reverse = 1;  // We need to gather positions
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

  if (comm->me == 0)
    std::cout << "[PAIR_IANN] Using cutoff: " << cutoff << " Angstrom" << std::endl;
  
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
  torch::Tensor positions_tensor = torch::zeros({nlocal, 3}, torch::kFloat64);
  auto coord_acc = positions_tensor.accessor<double, 2>();
  for (int i = 0; i < nlocal; i++) {
      coord_acc[i][0] = x[i][0];
      coord_acc[i][1] = x[i][1];
      coord_acc[i][2] = x[i][2];
  }

    // Convert box to tensor (3,3) for both orthogonal and non-orthogonal boxes
    torch::Tensor cell_tensor = torch::zeros({3,3}, torch::kFloat64);
    auto cell_acc = cell_tensor.accessor<double,2>();
    
    // Get box vectors
    double *boxlo = domain->boxlo;
    double *boxhi = domain->boxhi;
    double xy = domain->xy;
    double xz = domain->xz;
    double yz = domain->yz;
    
    // Set cell matrix elements
    cell_acc[0][0] = boxhi[0] - boxlo[0];  // a_x
    cell_acc[0][1] = 0.0;
    cell_acc[0][2] = 0.0;

    cell_acc[1][0] = xy;                  // b_x
    cell_acc[1][1] = boxhi[1] - boxlo[1]; // b_y
    cell_acc[1][2] = 0.0;

    cell_acc[2][0] = xz;                  // c_x
    cell_acc[2][1] = yz;                  // c_y
    cell_acc[2][2] = boxhi[2] - boxlo[2]; // c_z

  // Build edges based on neighborlist
  build_edges(list->inum, list->ilist, list->numneigh, list->firstneigh);

  // Debug: print input tensor sizes for sanity
  std::cout << "[PAIR_IANN] total_atoms=" << num_atoms_tensor << std::endl;
  std::cout << "[PAIR_IANN] atom_types=" << atomic_numbers_tensor << std::endl;
  std::cout << "[PAIR_IANN] atom_positions=" << positions_tensor << std::endl;
  std::cout << "[PAIR_IANN] cell=" << cell_tensor << std::endl;
  std::cout << "[PAIR_IANN] edge_indices=" << edge_indices_tensor << std::endl;
  std::cout << "[PAIR_IANN] edge_vectors=" << edge_vectors_tensor << std::endl;
  std::cout << "[PAIR_IANN] num_edges=" << num_edges_tensor << std::endl;
  
  // Inference: call the scripted model with raw tensor inputs matching wrapper signature
  try {
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
    
    // Debug: print energy value and forces tensor shape
    double e_val = energy.item<double>();
    auto f_sizes = forces.sizes();
    std::cout << "[PAIR_IANN] returned energy = " << e_val << std::endl;
    std::cout << "[PAIR_IANN] forces shape = [" << f_sizes[0] << ", " << f_sizes[1] << "]" << std::endl;
    
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
    if (eflag_global) eng_vdwl = energy.item<double>();
    
    // Per-atom energies (if model provides them)
    if (eflag_atom && output.find("atomic_energy") != output.end()) {
      auto atom_energies = output.at("atomic_energy").toTensor();
      auto energies_accessor = atom_energies.accessor<float, 1>();
      for (int i = 0; i < nlocal; i++) {
        eatom[i] = energies_accessor[i];
      }
    }
    
    if (vflag_fdotr) virial_fdotr_compute();

    if (atomic_numbers_tensor.size(0) != nlocal || positions_tensor.size(0) != nlocal) {
      error->all(FLERR, "Tensor shape mismatch with nlocal atoms");
    }
    if (torch::any(torch::isnan(forces)).item<bool>()) {
      error->all(FLERR, "[IANN] NaNs in predicted forces.");
    }
    if (forces.abs().max().item<float>() > 1e4) {
      error->all(FLERR, "[IANN] Unphysically large force detected.");
    }
  }
  catch (const c10::Error& e) {
    error->all(FLERR, "Error running ML model: " + std::string(e.what()));
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
  if (narg < 3) error->all(FLERR, "Illegal pair_style command: need model type and model path");
  
  int iarg = 0;
  
  // Get model type
  model_type = utils::strdup(arg[iarg++]);
  
  // Get model path
  model_path = utils::strdup(arg[iarg++]);
  
  // Get cutoff
  cutoff = utils::numeric(FLERR, arg[iarg++], false, lmp);
  
  // Load the model
  try {
    model = std::make_shared<torch::jit::Module>(torch::jit::load(model_path));
    model->eval();  // Set to inference mode
  }
  catch (const c10::Error& e) {
    error->all(FLERR, "Error loading ML model: " + std::string(e.what()));
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
  
  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
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
    error->all(FLERR, "All pair coeffs must be set for ML potentials");
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
      edge_vectors.push_back({dx, dy, dz});
    }
  }

  // Now create tensors
  int num_edges = edge_indices.size();
  edge_indices_tensor = torch::empty({num_edges, 2}, torch::kInt64);
  edge_vectors_tensor = torch::empty({num_edges, 3}, torch::kFloat64);

  auto edge_indices_accessor = edge_indices_tensor.accessor<int64_t, 2>();
  auto edge_vectors_accessor = edge_vectors_tensor.accessor<double, 2>();

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