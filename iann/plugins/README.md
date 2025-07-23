# IANN-LAMMPS Interface

This plugin provides a LAMMPS interface for using trained interatomic neural network (IANN) potentials including PaiNN, NequIP, MACE, and EquiformerV2 in molecular dynamics simulations.

## Installation

# Prerequisites

- LibTorch (PyTorch C++ API) v1.8.0 or later
- LAMMPS (with C++11 support or later)

# Install LAMMPS intergrated with libtorch 

TorchScript installation  from official website: https://pytorch.org/get-started/locally/

Here is an example that selects Stable, Linux, LibTorch, C++/Java, and CUDA 11.8. Then downloading as follows: 

   ```bash
   INSTALL_PATH=~/changzhi/softwares
   cd $INSTALL_PATH
   wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu118.zip
   unzip libtorch-cxx11-abi-shared-with-deps-2.7.1+cu118.zip
   cd libtorch
   ```

Lammps GPU version installation intergrated with libtorch:

   ```
   cd $INSTALL_PATH
   git clone https://github.com/lammps/lammps.git
   cd lammps/src
   cp $INSTALL_PATH/IANN/iann/plugins/*.h $INSTALL_PATH/IANN/iann/plugins/*.cpp .
   cd .. && mkdir build && cd build

   # GPU version
   module load PrgEnv-nvidia gcc cmake openmpi cudatoolkit # Load required modules on NERSC. It may be different on different servers
   GPU_ARCH=`nvidia-smi --query-gpu=name --format=csv,noheader`
   cmake ../cmake -DCMAKE_PREFIX_PATH=$INSTALL_PATH/libtorch \
   -DCMAKE_CXX_FLAGS="-I$INSTALL_PATH/libtorch/include/torch/csrc/api/include -I$INSTALL_PATH/libtorch/include"   \
   -DTorch_DIR=$INSTALL_PATH/libtorch/share/cmake/Torch \
   -DCMAKE_BUILD_TYPE=Release -DPKG_GPU=yes  -DGPU_API=cuda -DGPU_ARCH=$GPU_ARCH \
   -DPKG_USER-MISC=ON -DBUILD_MPI=ON   -DBUILD_OMP=ON   \
   -DCMAKE_EXE_LINKER_FLAGS="-L$INSTALL_PATH/libtorch/lib -Wl,-rpath,$INSTALL_PATH/libtorch/lib -ltorch \
   -ltorch_cpu -lc10" ; make -j 8
   ```

If you want to make CPU version LAMMPS with LibTorch rather than GPU version, you can use the following command:

   ```
   # CPU version
   module load GCC/11.3.0 CMake/3.23.1-GCCcore-11.3.0 OpenMPI  # Load required modules on S3DF. It requires GCC≥ 7.1, CMake≥ 3.18, OpenMP
   cmake ../cmake -DCMAKE_PREFIX_PATH=$INSTALL_PATH/libtorch \
   -DCMAKE_CXX_FLAGS="-I$INSTALL_PATH/libtorch/include/torch/csrc/api/include -I$INSTALL_PATH/libtorch/include" \
   -DTorch_DIR=$INSTALL_PATH/libtorch/share/cmake/Torch \
   -DPKG_USER-MISC=ON -DBUILD_MPI=ON -DBUILD_OMP=ON \
   -DCMAKE_EXE_LINKER_FLAGS="-L$INSTALL_PATH/libtorch/lib -Wl,-rpath,$INSTALL_PATH/libtorch/lib -ltorch \
   -ltorch_cpu -lc10" ; make -j 8
   ```


## Export your trained model

First, you need to have a trained model with torch format, which can be obtained by running the training script. Then convert the model to the torchscript format as follows:

   ```python
   from iann.plugins.converter import convert_model_for_lammps

   convert_model_for_lammps(model_path='model.pth', 
                           model_type='painn', # if not specified, the model type will be inferred from the model file
                           output_path='model_lmp.pth')
   ```

Replace `painn` with your model type (`nequip`, `mace`, or `equiformer2`).

Note: load environments before exporting if you are on NERSC:

   ```bash
   module purge
   module load PrgEnv-nvidia
   module load openmpi
   module load cudatoolkit/11.7
   export PYTHONPATH=/pscratch/sd/c/changzhi/softwares/IANN_v2/IANN:$PYTHONPATH
   ```

Note: always use same cuda version when export, compile and run, for example: module load cudatoolkit/11.7


## Usage

Here's a sample LAMMPS input script using the IANN pair style:

   ```
   # Input script for LAMMPS with IANN potentials

   units metal
   atom_style atomic
   boundary p p p

   read_data initial.data

   # Define the IANN pair style
   pair_style iann painn /path/to/painn_lammps.pt 5.5
   pair_coeff * * 

   # Run your simulation
   timestep 0.001
   fix 1 all nvt temp 300.0 300.0 0.1
   thermo 100
   dump 1 all custom 1 dump.xyz id type x y z   # dump every timestep to record full trajectory

   run 5000
   ```

### Pair Style Parameters

The `pair_style iann` command takes the following parameters:

   ```
   pair_style iann model_type model_path cutoff
   ```

- `model_type`: Type of ML model (painn, nequip, mace, equiformer2)
- `model_path`: Path to the exported TorchScript model
- `cutoff`: Interaction cutoff distance in Å (must match the model's trained cutoff)

## Notes on Performance

The IANN potential computation is more computationally intensive than classical force fields. For optimal performance:

1. Use GPU acceleration if available (requires LibTorch with CUDA)
2. Consider smaller simulation sizes or longer timesteps
3. Enable OpenMP if available by building LAMMPS with `-DBUILD_OMP=ON`

## Limitations

- Atomic types in LAMMPS must match those used during model training
- Currently supports only NVT and NVE ensembles
- Periodic boundary conditions are required

## Troubleshooting

If you encounter issues with loading the model, ensure:

1. The LibTorch version used for building LAMMPS matches the version used to export the model
2. LAMMPS was compiled with C++11 or later
3. The model was exported with the correct compute_forces=True flag
4. Ensure that `/path/to/libtorch` in your CMake command points to the actual LibTorch C++ installation directory (where `include/torch/torch.h` resides); otherwise you'll see a "torch/torch.h: No such file or directory" error.
5. If you haven't installed LibTorch yet, download the C++ API archive from https://pytorch.org/get-started/locally/, extract it to a local folder, and point CMAKE_PREFIX_PATH to that path.
6. If your headers live under `libtorch/include/torch/csrc/api/include/torch/torch.h` instead of `libtorch/include/torch/torch.h`, ensure your `CMAKE_PREFIX_PATH` points to the LibTorch root directory (the folder containing both `include/` and `lib/`). If CMake does not pick up the correct include paths, manually add the appropriate flags:

   ```bash
   cmake ../cmake \
      -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DCMAKE_CXX_FLAGS="-I/path/to/libtorch/include/torch/csrc/api/include -I/path/to/libtorch/include" \
      -DPKG_USER-MISC=ON -DBUILD_MPI=ON -DBUILD_OMP=ON
   ```

7. If you get an undefined reference to `ompi_mpi_double`, make sure you're building with the same Open MPI you load via `mpicc`/`mpicxx` (e.g. unload `cray-mpich` and `module load openmpi`) or explicitly set `-DMPI_C_COMPILER=$(which mpicc) -DMPI_CXX_COMPILER=$(which mpicxx)` so CMake finds and links the correct MPI library.

### Resolving persistent LibTorch linking issues

If you continue to see many undefined references to LibTorch/C10 symbols (`at::_ops::`, `c10::`, `torch::jit::`, etc.), try the following approaches:

1. **Patch LAMMPS CMakeLists.txt to find and link Torch**:
   
   ```bash
   # In your lammps/src/ directory, edit CMakeLists.txt to add:
   find_package(Torch REQUIRED)
   # And in the target_link_libraries section for the lmp target, add ${TORCH_LIBRARIES}
   ```

2. **Direct linking of LibTorch libraries**:
   
   ```bash
   # Create a file in your build directory called link_fix.cmake
   echo 'target_link_libraries(lmp PRIVATE 
      ${MPI_LIBRARIES}
      /global/homes/c/changzhi/changzhi/softwares/libtorch/lib/libtorch.so
      /global/homes/c/changzhi/changzhi/softwares/libtorch/lib/libtorch_cpu.so
      /global/homes/c/changzhi/changzhi/softwares/libtorch/lib/libc10.so
      ${LAMMPS_DEP_LIBS}
   )' > link_fix.cmake
   
   # Then run cmake with this include file
   cmake ../cmake -DCMAKE_PREFIX_PATH=/global/homes/c/changzhi/changzhi/softwares/libtorch \
     -DCMAKE_CXX_FLAGS="-I/global/homes/c/changzhi/changzhi/softwares/libtorch/include/torch/csrc/api/include -I/global/homes/c/changzhi/changzhi/softwares/libtorch/include" \
     -DTorch_DIR=/global/homes/c/changzhi/changzhi/softwares/libtorch/share/cmake/Torch \
     -DPKG_USER-MISC=ON -DBUILD_MPI=ON -DBUILD_OMP=ON \
     -DCMAKE_PROJECT_INCLUDE=link_fix.cmake
   ```

3. **Link against static LibTorch libraries**:

   ```bash
   # Link against the static libraries if available
   cmake ../cmake -DCMAKE_PREFIX_PATH=/global/homes/c/changzhi/changzhi/softwares/libtorch \
     -DCMAKE_CXX_FLAGS="-I/global/homes/c/changzhi/changzhi/softwares/libtorch/include/torch/csrc/api/include -I/global/homes/c/changzhi/changzhi/softwares/libtorch/include" \
     -DTorch_DIR=/global/homes/c/changzhi/changzhi/softwares/libtorch/share/cmake/Torch \
     -DPKG_USER-MISC=ON -DBUILD_MPI=ON -DBUILD_OMP=ON \
     -DTORCH_USE_STATIC_LIBS=ON
   ```

4. **Verify LibTorch installation**:

   ```bash
   # Check if the libraries exist and are readable
   ls -la /global/homes/c/changzhi/changzhi/softwares/libtorch/lib/lib*.so
   
   # Verify torch shared libraries with ldd
   ldd /global/homes/c/changzhi/changzhi/softwares/libtorch/lib/libtorch.so
   
   # Check if the libraries are loadable by the dynamic linker
   LD_LIBRARY_PATH=/global/homes/c/changzhi/changzhi/softwares/libtorch/lib ldconfig -p | grep torch
   ```
   
5. **Build a minimal C++ example to test LibTorch linking**:

   ```bash
   # Create test.cpp
   cat > test.cpp << 'EOF'
   #include <torch/torch.h>
   int main() {
     torch::Tensor tensor = torch::rand({2, 3});
     std::cout << tensor << std::endl;
     return 0;
   }
   EOF
   
   # Compile with similar flags
   g++ test.cpp -I/global/homes/c/changzhi/changzhi/softwares/libtorch/include \
      -I/global/homes/c/changzhi/changzhi/softwares/libtorch/include/torch/csrc/api/include \
      -L/global/homes/c/changzhi/changzhi/softwares/libtorch/lib \
      -Wl,-rpath,/global/homes/c/changzhi/changzhi/softwares/libtorch/lib \
      -ltorch -lc10 -ltorch_cpu -o test_torch
   
   # Run to confirm correct linking
   ./test_torch
   ```


## Known bugs:

### Bug 1:
   ```python
   File "/opt/anaconda3/lib/python3.11/site-packages/e3nn/nn/_activation.py", line 202
      index = 0
      # self.paths = [(mul, (l, p), act) for (mul, (l, p)), act in zip(self.irreps_in, self.acts)]
      for mul, (l, _), act in self.paths:
                              ~~~~~~~~~~ <--- HERE
         ir_dim = 2 * l + 1
         if act is not None:
   ```

### Solution 1:
   ```python
   import torch

   from e3nn import o3
   from e3nn.math import normalize2mom
   from e3nn.util.jit import compile_mode
   from e3nn.o3._irreps import Irreps

   @compile_mode("script")
   class Activation(torch.nn.Module):
      r"""Scalar activation function.

      Odd scalar inputs require activation functions with a defined parity (odd or even).

      Parameters
      ----------
      irreps_in : `e3nn.o3.Irreps`
         representation of the input

      acts : list of function or None
         list of activation functions, `None` if non-scalar or identity

      Examples
      --------

      >>> a = Activation("256x0o", [torch.abs])
      >>> a.irreps_out
      256x0e

      >>> a = Activation("256x0o+16x1e", [None, None])
      >>> a.irreps_out
      256x0o+16x1e
      """

      def __init__(self, irreps_in, acts) -> None:
         super().__init__()
         irreps_in = Irreps(irreps_in)
         if len(irreps_in) != len(acts):
               raise ValueError(f"Irreps in and number of activation functions does not match: {len(acts), (irreps_in, acts)}")

         # normalize the second moment
         acts = [normalize2mom(act) if act is not None else None for act in acts]

         from e3nn.util._argtools import _get_device

         irreps_out = []
         for (mul, (l_in, p_in)), act in zip(irreps_in, acts):
               if act is not None:
                  if l_in != 0:
                     raise ValueError("Activation: cannot apply an activation function to a non-scalar input.")

                  x = torch.linspace(0, 10, 256, device=_get_device(act))

                  a1, a2 = act(x), act(-x)
                  if (a1 - a2).abs().max() < 1e-5:
                     p_act = 1
                  elif (a1 + a2).abs().max() < 1e-5:
                     p_act = -1
                  else:
                     p_act = 0

                  p_out = p_act if p_in == -1 else p_in
                  irreps_out.append((mul, (0, p_out)))

                  if p_out == 0:
                     raise ValueError(
                           "Activation: the parity is violated! The input scalar is odd but the activation is neither "
                           "even nor odd."
                     )
               else:
                  irreps_out.append((mul, (l_in, p_in)))

         self.irreps_in = irreps_in
         self.irreps_out = Irreps(irreps_out)
         self.acts = torch.nn.ModuleList(acts)
         self.paths = [(mul, (l, p), act) for (mul, (l, p)), act in zip(self.irreps_in, self.acts)]
         assert len(self.irreps_in) == len(self.acts)

      def __repr__(self) -> str:
         acts = "".join(["x" if a is not None else " " for a in self.acts])
         return f"{self.__class__.__name__} [{acts}] ({self.irreps_in} -> {self.irreps_out})"

      def forward(self, features, dim: int = -1):
         """evaluate

         Parameters
         ----------
         features : `torch.Tensor`
               tensor of shape ``(...)``

         Returns
         -------
         `torch.Tensor`
               tensor of shape the same shape as the input
         """
         # - PROFILER - with torch.autograd.profiler.record_function(repr(self)):
         output = []
         index = 0
         for (mul, (l, p)), act in zip(self.irreps_in, self.acts): # Update this line, Fix torchscript error; Report on github: https://github.com/e3nn/e3nn/blob/0.5.6/e3nn/nn/_activation.py#L8-L111
               ir_dim = 2 * l + 1
               if act is not None:
                  output.append(act(features.narrow(dim, index, mul)))
               else:
                  output.append(features.narrow(dim, index, mul * ir_dim))
               index += mul * ir_dim

         if len(output) > 1:
               return torch.cat(output, dim=dim)
         elif len(output) == 1:
               return output[0]
         else:
               return torch.zeros_like(features)

   ```


### Bug 2:

## type error
   ```python
   normalize(Tensor input, float p=2., int dim=1, float eps=9.9999999999999998e-13, Tensor? out=None) -> Tensor:
   Expected a value of type 'float' for argument 'p' but instead found type 'int'.
   File "/global/homes/c/changzhi/softwares/conda/lib/python3.11/site-packages/e3nn/o3/_rotation.py", line 693
   'SO3_Rotation.RotationToWignerDMatrix' is being compiled since it was called from 'SO3_Rotation.set_wigner'
   File "/pscratch/sd/c/changzhi/softwares/IANN_v2/IANN/iann/models/equiformerV2.py", line 351
         # self.device, self.dtype = rot_mat3x3.device, rot_mat3x3.dtype
         length = len(rot_mat3x3)  
         self.wigner = self.RotationToWignerDMatrix(rot_mat3x3, 0, self.lmax)
         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
         self.wigner_inv = torch.transpose(self.wigner, 1, 2).contiguous()
         self.wigner = self.wigner.detach()
   ```

## Solution 2:

   ```python
   def xyz_to_angles(xyz):
      r"""convert a point :math:`\vec r = (x, y, z)` on the sphere into angles :math:`(\alpha, \beta)`

      .. math::

         \vec r = R(\alpha, \beta, 0) \vec e_z


      Parameters
      ----------
      xyz : `torch.Tensor`
         tensor of shape :math:`(..., 3)`

      Returns
      -------
      alpha : `torch.Tensor`
         tensor of shape :math:`(...)`

      beta : `torch.Tensor`
         tensor of shape :math:`(...)`

      # site-packages/e3nn/o3/_rotation.py
      """
      xyz = torch.nn.functional.normalize(xyz, p=2.0, dim=-1)  # forward 0's instead of nan for zero-radius, updated 2 -> 2.0
      xyz = xyz.clamp(-1, 1)

      beta = torch.acos(xyz[..., 1])
      alpha = torch.atan2(xyz[..., 0], xyz[..., 2])
      return alpha, beta
    ```

