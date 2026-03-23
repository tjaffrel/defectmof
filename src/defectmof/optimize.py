"""Geometry optimization using MACE force fields."""

from ase import Atoms
from ase.io import write

from defectmof._ase import ase_optimize


def optimize(
    atoms: str | Atoms,
    model: str = "mace_mp_medium",
    optimizer: str = "lbfgs",
    fmax: float = 0.05,
    max_steps: int = 500,
    backend: str = "ase",
    device: str = "cuda",
    memory_limit_gb: float | None = None,
    output: str | None = None,
    head: str | None = None,
) -> Atoms:
    """Optimize a structure using MACE.

    Args:
        atoms: Path to CIF file or ASE Atoms object.
        model: MACE model name ("mace_mp_small", "mace_mp_medium") or path to .model file.
        optimizer: Optimization algorithm ("lbfgs", "fire", "bfgs").
        fmax: Force convergence criterion in eV/A.
        max_steps: Maximum optimization steps.
        backend: Computation backend ("ase" or "torchsim").
        device: Compute device ("cuda" or "cpu").
        memory_limit_gb: Max GPU memory for torchsim backend.
        output: If set, save optimized structure to this CIF path.
        head: For multi-head MACE models, which head to use (e.g. "pt_head").

    Returns:
        Optimized ASE Atoms object.
    """
    if fmax <= 0:
        raise ValueError(f"fmax must be positive, got {fmax}")
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")

    if optimizer not in ("lbfgs", "fire", "bfgs"):
        raise ValueError(f"Unknown optimizer '{optimizer}'. Use: 'lbfgs', 'fire', 'bfgs'")

    if backend == "ase":
        result = ase_optimize(atoms, model, optimizer, fmax, max_steps, device, head=head)
    elif backend == "torchsim":
        from defectmof._torchsim import torchsim_optimize
        result = torchsim_optimize(atoms, model, optimizer, fmax, max_steps, device, memory_limit_gb)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use: 'ase' or 'torchsim'")

    if output:
        write(output, result)

    return result
