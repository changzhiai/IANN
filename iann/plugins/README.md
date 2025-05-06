# IANN-LAMMPS Interface

This package provides a LAMMPS interface for using trained interatomic neural network (IANN) potentials including PaiNN, NEQuIP, MACE, and EquiformerV2 in molecular dynamics simulations.

## Prerequisites

- LibTorch (PyTorch C++ API) v1.8.0 or later
- LAMMPS (with C++11 support or later)

## Installation

### 1. Export your trained model

First, export your trained model to TorchScript format:

```bash
python export_model.py path/to/best_model.pth painn --output painn_lammps.pt
```

Replace `painn` with your model type (`nequip`, `mace`, or `equiformer2`).

### 2. Build LAMMPS with LibTorch and the IANN package

1. Download and install LibTorch from https://pytorch.org/get-started/locally/

2. Copy `pair_iann.h` and `pair_iann.cpp` to your LAMMPS source directory:

```bash
cp pair_iann.h pair_iann.cpp /path/to/lammps/src
```

#### Building on NERSC

```bash
# Load required modules
module load PrgEnv-nvidia gcc openmpi cmake cudatoolkit # use Open MPI to satisfy ompi_mpi_double symbol

# Clone LAMMPS repository and copy the IANN plugin
git clone https://github.com/lammps/lammps.git ~/lammps
cp pair_iann.h pair_iann.cpp ~/lammps/src/

# Create and enter build directory
mkdir -p ~/lammps/build && cd ~/lammps/build

# Configure with LibTorch (update LIBTORCH_PATH)


cmake ../cmake   -DCMAKE_PREFIX_PATH=/global/homes/c/changzhi/changzhi/softwares/libtorch \
  -DCMAKE_CXX_FLAGS="-I/global/homes/c/changzhi/changzhi/softwares/libtorch/include/torch/csrc/api/include -I/global/homes/c/changzhi/changzhi/softwares/libtorch/include"   \
  -DTorch_DIR=/global/homes/c/changzhi/changzhi/softwares/libtorch/share/cmake/Torch \
  -DPKG_USER-MISC=ON   -DBUILD_MPI=ON   -DBUILD_OMP=ON   \
  -DCMAKE_EXE_LINKER_FLAGS="-L/global/homes/c/changzhi/changzhi/softwares/libtorch/lib -Wl,-rpath,/global/homes/c/changzhi/changzhi/softwares/libtorch/lib -ltorch \
  -ltorch_cpu -lc10" ;
make -j 8
```

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

#### Patching LAMMPS CMakeLists for Full Torch Support

If undefined references persist, ensure the LAMMPS build system explicitly finds and links the full set of Torch libraries via CMake's `find_package`. In `lammps/src/CMakeLists.txt`, apply a patch like:

```diff
 find_package(MPI REQUIRED)
+find_package(Torch REQUIRED PATHS /global/homes/c/changzhi/changzhi/softwares/libtorch/share/cmake/Torch)
+include_directories(${TORCH_INCLUDE_DIRS})
+set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
@@
 add_executable(lmp ${srcs} ${hdrs})
 target_link_libraries(lmp
     ${MPI_LIBRARIES}
+    ${TORCH_LIBRARIES}
     ${LAMMPS_DEP_LIBS}
 )
```

Then reconfigure and rebuild in your build directory:

```bash
cd ~/lammps/build
cmake ../cmake \
  -DCMAKE_PREFIX_PATH=/global/homes/c/changzhi/changzhi/softwares/libtorch \
  -DPKG_USER-MISC=ON -DBUILD_MPI=ON -DBUILD_OMP=ON
make -j${SLURM_CPUS_ON_NODE:-$(nproc)}
```

This leverages CMake's Torch support to pull in and link all required (`c10`, `torch`, JIT, CPU/CUDA backends, etc.) libraries automatically. 