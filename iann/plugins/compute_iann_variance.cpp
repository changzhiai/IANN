/* ----------------------------------------------------------------------
   Custom thermo style for IANN potential

   There are 4 components to the variance:
   - energy_variance
   - force_variance
   - max_energy_variance
   - max_force_variance
   
   # Usage: 
   
   # Set up compute for variance
   compute variance all iann/variance

   # Use in thermo output
   thermo_style custom step pe ke etotal press c_variance[1] c_variance[2] c_variance[3] c_variance[4]

   thermo_modify colname c_variance[1] energy_variance
   thermo_modify colname c_variance[2] force_variance
   thermo_modify colname c_variance[3] max_energy_variance
   thermo_modify colname c_variance[4] max_force_variance

------------------------------------------------------------------------- */

#include "compute_iann_variance.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "update.h"
#include "pair_iann.h"
#include "pair_iann_multi_gpu_variance.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeIANNVariance::ComputeIANNVariance(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR, "Illegal compute iann/variance command");

  vector_flag = 1;
  size_vector = 4;
  extvector = 0;

  vector = new double[4];

  // Get pointer to IANN pair style
  pair_iann = (PairIANN *)force->pair_match("iann", 1);
  pair_iann_multi_gpu = (PairIANNMultiGPUVariance *)force->pair_match("iann/multi_gpu/variance", 1);
  
  if (!pair_iann && !pair_iann_multi_gpu) {
    error->all(FLERR, "Compute iann/variance requires pair style iann or iann/multi_gpu/variance");
  }
}

/* ---------------------------------------------------------------------- */

ComputeIANNVariance::~ComputeIANNVariance()
{
  delete[] vector;
}

/* ---------------------------------------------------------------------- */

void ComputeIANNVariance::init()
{
  // Check if pair style is still available
  pair_iann = (PairIANN *)force->pair_match("iann", 1);
  pair_iann_multi_gpu = (PairIANNMultiGPUVariance *)force->pair_match("iann/multi_gpu/variance", 1);
  
  if (!pair_iann && !pair_iann_multi_gpu) {
    error->all(FLERR, "Compute iann/variance requires pair style iann or iann/multi_gpu/variance");
  }
}

/* ---------------------------------------------------------------------- */

void ComputeIANNVariance::compute_vector()
{
  // Access variance values from pair style
  if (pair_iann) {
    vector[0] = pair_iann->energy_variance;
    vector[1] = pair_iann->force_variance;
    vector[2] = pair_iann->max_energy_variance;
    vector[3] = pair_iann->max_force_variance;
  } else if (pair_iann_multi_gpu) {
    vector[0] = pair_iann_multi_gpu->energy_variance;
    vector[1] = pair_iann_multi_gpu->force_variance;
    vector[2] = pair_iann_multi_gpu->max_energy_variance;
    vector[3] = pair_iann_multi_gpu->max_force_variance;
  }
}

/* ---------------------------------------------------------------------- */

double ComputeIANNVariance::memory_usage()
{
  double bytes = 4 * sizeof(double);  // Size of vector array
  return bytes;
} 