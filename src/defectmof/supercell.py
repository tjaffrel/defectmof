"""Supercell building with controlled defect distributions."""

import numpy as np
from ase import Atoms
from ase.io import read

from defectmof.utils import fix_short_cbr_bonds, prune_overlapping_atoms


def build_supercell(
    defective: str | Atoms,
    pristine: str | Atoms,
    size: tuple = (4, 4, 4),
    defect_fraction: float = 0.44,
    mode: str = "random",
    seed: int = 42,
) -> Atoms:
    """Build a mixed defective/pristine MOF supercell.

    Args:
        defective: Path to CIF file or ASE Atoms for the defective unit cell.
        pristine: Path to CIF file or ASE Atoms for the pristine unit cell.
        size: Supercell dimensions (nx, ny, nz).
        defect_fraction: Fraction of unit cells that are defective (0.0 to 1.0).
        mode: Arrangement mode. One of: "random", "alternating_a", "alternating_b",
            "alternating_c", "alternating_ab", "alternating_ac", "alternating_bc",
            "alternating_abc", "clustered_small", "clustered_large".
        seed: Random seed for reproducibility.

    Returns:
        ASE Atoms object with the assembled supercell.

    Raises:
        ValueError: If cell parameters don't match or invalid mode.
        FileNotFoundError: If CIF file path doesn't exist.
    """
    if not isinstance(size, (tuple, list)) or len(size) != 3:
        raise ValueError(f"size must be a tuple of 3 integers, got {size}")
    if any(s < 1 for s in size):
        raise ValueError(f"All size dimensions must be >= 1, got {size}")
    if not 0.0 <= defect_fraction <= 1.0:
        raise ValueError(f"defect_fraction must be between 0.0 and 1.0, got {defect_fraction}")

    if isinstance(defective, str):
        defective = read(defective)
    if isinstance(pristine, str):
        pristine = read(pristine)

    _validate_cells(defective, pristine)

    defective = defective.copy()
    defective.set_cell(pristine.get_cell(), scale_atoms=False)

    nx, ny, nz = size
    grid_labels = _generate_arrangements(nx, ny, nz, defect_fraction, mode, seed)

    supercell = Atoms(cell=pristine.get_cell() * [nx, ny, nz], pbc=True)
    vec_a, vec_b, vec_c = pristine.get_cell()

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                chosen = defective if grid_labels[i, j, k] == 1 else pristine
                block = chosen.copy()
                block.translate(i * vec_a + j * vec_b + k * vec_c)
                supercell.extend(block)

    supercell = prune_overlapping_atoms(supercell, threshold=0.8)
    supercell = fix_short_cbr_bonds(supercell, target_dist=2.06)

    return supercell


def _validate_cells(cell_a: Atoms, cell_b: Atoms) -> None:
    """Check that two unit cells have compatible cell parameters."""
    params_a = cell_a.cell.cellpar()
    params_b = cell_b.cell.cellpar()

    length_tol = 0.01
    angle_tol = 0.1

    for i, (a, b) in enumerate(zip(params_a, params_b)):
        tol = length_tol if i < 3 else angle_tol
        name = ["a", "b", "c", "alpha", "beta", "gamma"][i]
        if abs(a - b) > tol:
            raise ValueError(
                f"Mismatched cell parameters: {name} differs by {abs(a-b):.4f} "
                f"(defective={a:.4f}, pristine={b:.4f}). "
                f"Tolerance: {tol}. Both cells must have matching lattice parameters."
            )


def _generate_arrangements(
    nx: int, ny: int, nz: int,
    defect_fraction: float,
    mode: str,
    seed: int,
) -> np.ndarray:
    """Generate a 3D grid labeling each cell as defective (1) or pristine (2)."""
    rng = np.random.RandomState(seed)
    total = nx * ny * nz
    num_defective = int(round(defect_fraction * total))

    pattern = np.zeros((nx, ny, nz))

    valid_modes = {
        "random", "alternating_a", "alternating_b", "alternating_c",
        "alternating_ab", "alternating_ac", "alternating_bc", "alternating_abc",
        "clustered_small", "clustered_large",
    }
    if mode not in valid_modes:
        raise ValueError(f"Unknown mode '{mode}'. Valid modes: {sorted(valid_modes)}")

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if mode == "alternating_a":
                    pattern[i, j, k] = 1 if i % 2 == 0 else 0
                elif mode == "alternating_b":
                    pattern[i, j, k] = 1 if j % 2 == 0 else 0
                elif mode == "alternating_c":
                    pattern[i, j, k] = 1 if k % 2 == 0 else 0
                elif mode == "alternating_ab":
                    pattern[i, j, k] = 1 if (i + j) % 2 == 0 else 0
                elif mode == "alternating_ac":
                    pattern[i, j, k] = 1 if (i + k) % 2 == 0 else 0
                elif mode == "alternating_bc":
                    pattern[i, j, k] = 1 if (j + k) % 2 == 0 else 0
                elif mode == "alternating_abc":
                    pattern[i, j, k] = 1 if (i + j + k) % 2 == 0 else 0
                elif mode == "clustered_small":
                    block_id = (i // 2) + (j // 2) * (nx // 2 + 1) + (k // 2) * (nx // 2 + 1) * (ny // 2 + 1)
                    pattern[i, j, k] = rng.rand() + block_id * 0.01
                elif mode == "clustered_large":
                    half_x, half_y, half_z = nx // 2, ny // 2, nz // 2
                    block_id = (i // max(half_x, 1)) + (j // max(half_y, 1)) * 2 + (k // max(half_z, 1)) * 4
                    pattern[i, j, k] = rng.rand() + block_id * 0.1
                else:
                    pattern[i, j, k] = rng.rand()

    flat = pattern.flatten()
    noise = rng.rand(total) * 1e-6
    ranked = np.argsort(-(flat + noise))

    labels = np.full(total, 2)
    labels[ranked[:num_defective]] = 1
    return labels.reshape((nx, ny, nz))
