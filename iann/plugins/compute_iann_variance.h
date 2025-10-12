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

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(iann/variance,ComputeIANNVariance);
// clang-format on
#else

#ifndef LMP_COMPUTE_IANN_VARIANCE_H
#define LMP_COMPUTE_IANN_VARIANCE_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeIANNVariance : public Compute {
 public:
  ComputeIANNVariance(class LAMMPS *, int, char **);
  ~ComputeIANNVariance() override;
  void init() override;
  void compute_vector() override;
  double memory_usage() override;

 private:
  class PairIANN *pair_iann;
  class PairIANNMultiGPU *pair_iann_multi_gpu;
};

}    // namespace LAMMPS_NS

#endif
#endif 