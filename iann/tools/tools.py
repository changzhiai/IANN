import logging
import numpy as np
from typing import List, Optional, Dict, Any

logger = logging.getLogger("iann.tools")


def calc_energy_statistics(
    images,
    species: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    seed: int = 42,
    log_file: Optional[str] = "energy_statistics.log",
) -> Dict[str, Any]:
    """Compute per-type energy shifts and scales from training data.
    
    Designed to scale to millions of structures via:
    - Random sampling (controlled by ``max_samples``)
    - Vectorized composition counting (``np.unique``)
    - Running sums for force RMS (O(1) memory per element)
    
    **Shifts** are computed via least-squares regression:
        E_total ≈ Σ_i  n_i * shift_i
    
    **Scales** are computed from per-element force RMS:
        scale_i = sqrt(mean(|F|^2)) for all atoms of type i
    
    Args:
        images: List of ASE Atoms objects with calculated energies 
            (and optionally forces). Can also be an ASE Trajectory path (str).
        species: List of element symbols (e.g., ['Si', 'O']).
            If None, auto-discovered from the dataset.
        max_samples: Maximum number of structures to use. If the dataset is
            larger, a random subset is sampled. None = use all (default).
        seed: Random seed for reproducible sampling.
        log_file: Path to save log output. Set to None to disable file logging.
            Default: 'energy_statistics.log'.
    
    Returns:
        Dictionary with:
            - 'per_type_energy_shifts': Dict[str, float]
            - 'per_type_energy_scales': Dict[str, float] (None if no forces)
            - 'avg_energy_per_atom': float
            - 'energy_std_per_atom': float
            - 'species': List[str] (sorted by atomic number)
    
    Example:
        >>> from ase.io import read
        >>> images = read('train.traj', index=':')
        >>> # Auto-discover species and sample 50k structures:
        >>> stats = calc_energy_statistics(
        ...     images, max_samples=50000
        ... )
    """
    import time
    from ase.io import Trajectory as AseTrajectory
    from ase.data import atomic_numbers as ase_atomic_numbers
    from ase.data import chemical_symbols as ase_chemical_symbols
    
    # Set up logging: console + optional file
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console)
    if log_file is not None:
        # Remove any previous file handlers to avoid duplicates
        logger.handlers = [h for h in logger.handlers 
                           if not isinstance(h, logging.FileHandler)]
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
    
    t0 = time.time()
    
    # Load trajectory if path given
    if isinstance(images, str):
        images = AseTrajectory(images)
    
    n_total = len(images)
    
    # Auto-discover species if not provided
    if species is None:
        logger.info("Auto-discovering species from dataset...")
        all_z = set()
        # Scan a subset for speed (use max_samples if given, else all)
        scan_limit = min(n_total, max_samples) if max_samples else n_total
        scan_indices = np.random.RandomState(seed).choice(
            n_total, size=scan_limit, replace=False
        ) if n_total > scan_limit else range(n_total)
        for idx in scan_indices:
            atoms = images[int(idx)]
            all_z.update(atoms.get_atomic_numbers().tolist())
        species = [ase_chemical_symbols[z] for z in sorted(all_z)]
        logger.info(f"  Found {len(species)} species: {species}")
    
    # Sort species by atomic number (matching TypeMapper convention)
    numbers = [ase_atomic_numbers[s] for s in species]
    sorted_species = [e[1] for e in sorted(zip(numbers, species))]
    num_types = len(sorted_species)
    
    # Build Z -> type index lookup (fast, avoids string comparisons)
    z_to_type = {}
    for idx, sp in enumerate(sorted_species):
        z_to_type[ase_atomic_numbers[sp]] = idx
    
    # Random sampling for large datasets
    if max_samples is not None and n_total > max_samples:
        rng = np.random.RandomState(seed)
        sample_indices = set(rng.choice(n_total, size=max_samples, replace=False))
        logger.info(f"Sampling {max_samples:,} / {n_total:,} structures (seed={seed})")
    else:
        sample_indices = None  # use all
    
    # === Single pass: collect compositions, energies, and force running sums ===
    compositions = []
    energies = []
    total_atoms = 0
    
    # Running sums for force RMS (O(1) memory per element)
    force_sq_sum = np.zeros(num_types)   # sum of |F|^2
    force_count = np.zeros(num_types, dtype=np.int64)  # number of atoms
    has_forces = False
    
    log_interval = max(1, n_total // 20)  # progress every 5%
    
    for i, atoms in enumerate(images):
        # Skip if not in sample
        if sample_indices is not None and i not in sample_indices:
            continue
        
        # Progress
        if i % log_interval == 0 and i > 0:
            pct = 100 * i / n_total
            elapsed = time.time() - t0
            logger.info(f"  [{pct:5.1f}%] Processed {i:,} / {n_total:,} "
                        f"({elapsed:.1f}s elapsed)")
        
        # Energy
        try:
            energy = atoms.get_potential_energy()
        except (AttributeError, RuntimeError):
            continue
        
        # Composition via vectorized np.unique on atomic numbers
        atom_numbers = atoms.get_atomic_numbers()
        comp = np.zeros(num_types)
        unique_z, counts = np.unique(atom_numbers, return_counts=True)
        for z, c in zip(unique_z, counts):
            if z in z_to_type:
                comp[z_to_type[z]] = c
        
        compositions.append(comp)
        energies.append(energy)
        total_atoms += len(atoms)
        
        # Forces — running sum, no per-atom storage
        try:
            forces = atoms.get_forces(apply_constraint=False)
            has_forces = True
            # Compute |F|^2 per atom = sum over xyz components
            f_sq = np.sum(forces ** 2, axis=1)  # [n_atoms]
            for z, fsq in zip(atom_numbers, f_sq):
                if z in z_to_type:
                    tid = z_to_type[z]
                    force_sq_sum[tid] += fsq
                    force_count[tid] += 1
        except (AttributeError, RuntimeError):
            pass
    
    n_used = len(compositions)
    if n_used == 0:
        raise ValueError("No structures with energies found in dataset!")
    
    A = np.array(compositions)  # [n_used, num_types]
    b = np.array(energies)      # [n_used]
    
    # Least-squares solve: A @ shifts = b
    shifts, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    per_type_shifts = {sp: float(shifts[i]) for i, sp in enumerate(sorted_species)}
    
    # Per-atom energy statistics
    atoms_per_struct = A.sum(axis=1)
    energy_per_atom = b / atoms_per_struct
    avg_e_per_atom = float(np.mean(energy_per_atom))
    std_e_per_atom = float(np.std(energy_per_atom))
    
    # === Force RMS scales ===
    per_type_scales = None
    if has_forces:
        per_type_scales = {}
        all_rms = []
        for i, sp in enumerate(sorted_species):
            if force_count[i] > 0:
                rms = float(np.sqrt(force_sq_sum[i] / force_count[i]))
            else:
                rms = 1.0
            per_type_scales[sp] = rms
            all_rms.append(rms)
        
        # Normalize so mean scale = 1.0
        mean_rms = np.mean(all_rms)
        if mean_rms > 0:
            per_type_scales = {sp: v / mean_rms for sp, v in per_type_scales.items()}
    
    elapsed = time.time() - t0
    
    # Log summary
    logger.info("=" * 60)
    logger.info("Per-Type Energy Statistics")
    logger.info("=" * 60)
    logger.info(f"  Dataset: {n_used:,} structures used"
                f" (of {n_total:,} total), {total_atoms:,} atoms")
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info(f"  Avg energy/atom: {avg_e_per_atom:.6f}")
    logger.info(f"  Std energy/atom: {std_e_per_atom:.6f}")
    logger.info(f"  Per-type energy shifts (from linear regression):")
    for sp, val in per_type_shifts.items():
        n = int(A[:, sorted_species.index(sp)].sum())
        logger.info(f"    {sp:>3s}: {val:12.6f}  ({n:,} atoms)")
    if per_type_scales is not None:
        logger.info(f"  Per-type energy scales (from force RMS, normalized):")
        for i, sp in enumerate(sorted_species):
            logger.info(f"    {sp:>3s}: {per_type_scales[sp]:12.6f}"
                        f"  ({int(force_count[i]):,} atoms)")
    else:
        logger.info(f"  Per-type energy scales: N/A (no forces in dataset)")
    logger.info("=" * 60)
    
    if log_file is not None:
        logger.info(f"  Log saved to: {log_file}")
    
    return {
        'per_type_energy_shifts': per_type_shifts,
        'per_type_energy_scales': per_type_scales,
        'avg_energy_per_atom': avg_e_per_atom,
        'energy_std_per_atom': std_e_per_atom,
        'species': sorted_species,
    }
