/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
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
  if (!pair_iann) error->all(FLERR, "Compute iann/variance requires pair style iann");
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
  if (!pair_iann) error->all(FLERR, "Compute iann/variance requires pair style iann");
}

/* ---------------------------------------------------------------------- */

void ComputeIANNVariance::compute_vector()
{
  // Access variance values from pair style
  vector[0] = pair_iann->energy_variance;
  vector[1] = pair_iann->force_variance;
  vector[2] = pair_iann->max_energy_variance;
  vector[3] = pair_iann->max_force_variance;
}

/* ---------------------------------------------------------------------- */

double ComputeIANNVariance::memory_usage()
{
  double bytes = 4 * sizeof(double);  // Size of vector array
  return bytes;
} 