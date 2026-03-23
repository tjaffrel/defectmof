"""Hierarchical optimization pipeline for large supercells."""

import numpy as np
from ase import Atoms
from ase.build import make_supercell
from ase.io import write

from defectmof.supercell import build_supercell
from defectmof.optimize import optimize


def hierarchical_optimize(
    defective: str | Atoms,
    pristine: str | Atoms,
    target_size: tuple = (8, 8, 8),
    defect_fraction: float = 0.44,
    mode: str = "random",
    model: str = "mace_mp_medium",
    backend: str = "ase",
    device: str = "cuda",
    output: str | None = None,
) -> Atoms:
    """Optimize in stages: 2x2x2 -> tile to target_size -> re-optimize.

    Args:
        defective: Path to CIF or ASE Atoms for defective unit cell.
        pristine: Path to CIF or ASE Atoms for pristine unit cell.
        target_size: Final supercell dimensions. Must be >= (4,4,4) and even.
        defect_fraction: Fraction of defective cells.
        mode: Arrangement mode.
        model: MACE model name or path.
        backend: "ase" or "torchsim".
        device: "cuda" or "cpu".
        output: Save final structure to this CIF path.

    Returns:
        Optimized ASE Atoms at target_size.
    """
    for i, dim in enumerate(target_size):
        axis = ["x", "y", "z"][i]
        if dim < 4:
            raise ValueError(
                f"target_size {axis}={dim} must be >= 4. "
                "Use optimize() directly for smaller supercells."
            )
        if dim % 2 != 0:
            raise ValueError(
                f"target_size {axis}={dim} must be divisible by 2."
            )

    # Stage 1: build and optimize at 2x2x2 (loose fmax)
    print("Stage 1: building 2x2x2 supercell...")
    sc = build_supercell(
        defective, pristine,
        size=(2, 2, 2),
        defect_fraction=defect_fraction,
        mode=mode,
    )
    print(f"  Optimizing ({len(sc)} atoms, fmax=0.1)...")
    optimized_small = optimize(sc, model=model, fmax=0.1, backend=backend, device=device)

    # Stage 2: tile optimized 2x2x2 into target_size and re-optimize
    tile_factors = tuple(t // 2 for t in target_size)
    print(f"Stage 2: tiling optimized 2x2x2 by {tile_factors} to reach {target_size}...")
    tiled = make_supercell(optimized_small, np.diag(tile_factors))

    print(f"  Optimizing ({len(tiled)} atoms, fmax=0.05)...")
    result = optimize(tiled, model=model, fmax=0.05, backend=backend, device=device)

    if output and result is not None:
        write(output, result)

    return result
