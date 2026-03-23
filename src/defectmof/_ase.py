"""ASE backend wrappers for optimization and MD."""

import os
import warnings

from ase import Atoms, units
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS, FIRE, LBFGS


def _load_atoms(atoms: str | Atoms) -> Atoms:
    """Load atoms from file path or return as-is.

    Supports CIF, XYZ, POSCAR, and other ASE-readable formats.
    For .cif files, forces CIF format to avoid misdetection (e.g. CONTCAR*.cif).
    """
    if isinstance(atoms, str):
        if not os.path.isfile(atoms):
            raise FileNotFoundError(
                f"Cannot read file: {atoms}. Check the file exists and is valid."
            )
        # Force CIF format for .cif files to avoid ASE misdetecting
        # filenames containing "CONTCAR" or "POSCAR" as VASP format
        if atoms.lower().endswith(".cif"):
            return read(atoms, format="cif")
        return read(atoms)
    return atoms.copy()


def _get_mace_calculator(
    model: str, device: str, head: str | None = None, dispersion: bool = False,
):
    """Load a MACE calculator with CUDA fallback to CPU.

    Args:
        model: Model name ("mace_mp_small", "mace_mp_medium") or path to .model file.
        device: "cuda" or "cpu".
        head: For multi-head models, specify which head to use (e.g. "pt_head", "pbe_d3").
        dispersion: Add D3(BJ) dispersion correction. Only for MACE-MP0 models
            (PBE+U level). Do NOT use with head="pbe_d3" — that already includes D3.
    """
    import torch
    from mace.calculators import MACECalculator, mace_mp

    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU. This will be slower.")
        device = "cpu"

    model_map = {"mace_mp_small": "small", "mace_mp_medium": "medium"}

    try:
        if model in model_map:
            # Standard MACE-MP0 model — supports built-in dispersion
            return mace_mp(
                model=model_map[model], device=device,
                default_dtype="float32", dispersion=dispersion,
            )
        else:
            # Custom model file — use MACECalculator directly
            kwargs = dict(model_paths=model, device=device, default_dtype="float32")
            if head is not None:
                kwargs["head"] = head
            if dispersion and head != "pbe_d3":
                warnings.warn(
                    "dispersion=True with custom models is not supported. "
                    "Use head='pbe_d3' for built-in D3 dispersion instead."
                )
            return MACECalculator(**kwargs)
    except Exception as e:
        if "urlopen" in str(e).lower() or "connection" in str(e).lower():
            raise ConnectionError(
                f"Cannot download MACE model '{model}'. "
                "Check your internet connection or provide a local model path."
            ) from e
        raise


def ase_optimize(
    atoms: str | Atoms,
    model: str,
    optimizer: str,
    fmax: float,
    max_steps: int,
    device: str,
    logfile: str | None = None,
    head: str | None = None,
    dispersion: bool = False,
) -> Atoms:
    """Run geometry optimization using ASE."""
    import torch

    atoms = _load_atoms(atoms)
    atoms.calc = _get_mace_calculator(model, device, head=head, dispersion=dispersion)

    opt_map = {"lbfgs": LBFGS, "fire": FIRE, "bfgs": BFGS}
    if optimizer not in opt_map:
        raise ValueError(f"Unknown optimizer '{optimizer}'. Use: {list(opt_map.keys())}")

    try:
        opt = opt_map[optimizer](atoms, logfile=logfile)
        converged = opt.run(fmax=fmax, steps=max_steps)

        if not converged:
            current_fmax = float(abs(atoms.get_forces()).max())
            warnings.warn(
                f"Optimization did not converge in {max_steps} steps "
                f"(fmax={current_fmax:.4f}). Returning best result."
            )
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError(
            "GPU out of memory during optimization. Try:\n"
            "  - A smaller supercell size\n"
            "  - backend='torchsim' with memory_limit_gb parameter\n"
            "  - device='cpu' (slower but no VRAM limit)"
        )

    return atoms


def ase_run_md(
    atoms: str | Atoms,
    temperature: float,
    n_steps: int,
    timestep: float,
    model: str,
    thermostat: str,
    friction: float,
    log_interval: int,
    equilibration_steps: int,
    device: str,
    trajectory: str | None,
    head: str | None = None,
    dispersion: bool = False,
) -> list[Atoms]:
    """Run MD simulation using ASE."""
    import torch
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    atoms = _load_atoms(atoms)
    atoms.calc = _get_mace_calculator(model, device, head=head, dispersion=dispersion)

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

    if thermostat == "langevin":
        dyn = Langevin(
            atoms,
            timestep=timestep * units.fs,
            temperature_K=temperature,
            friction=friction,
        )
    elif thermostat == "nose_hoover":
        from ase.md.nosehoover import NoseHoover
        dyn = NoseHoover(
            atoms,
            timestep=timestep * units.fs,
            temperature_K=temperature,
            ttime=25 * units.fs,
        )
    else:
        raise ValueError(f"Unknown thermostat '{thermostat}'. Use: 'langevin' or 'nose_hoover'")

    snapshots = []
    traj_writer = None
    if trajectory:
        traj_writer = Trajectory(trajectory, "w", atoms)

    try:
        total_steps = equilibration_steps + n_steps
        for step in range(1, total_steps + 1):
            dyn.run(1)
            if step > equilibration_steps and step % log_interval == 0:
                snapshots.append(atoms.copy())
                if traj_writer:
                    traj_writer.write(atoms)
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError(
            "GPU out of memory during MD. Try a smaller system or use torchsim backend."
        )
    finally:
        if traj_writer:
            traj_writer.close()

    return snapshots
