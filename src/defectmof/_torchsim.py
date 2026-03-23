"""torchsim backend wrappers with memory-safe defaults."""

import warnings

from ase import Atoms
from ase.io import read


def _check_torchsim():
    """Check if torchsim is installed."""
    try:
        import torch_sim  # noqa: F401
        return True
    except ImportError:
        raise ImportError(
            "torchsim is not installed. Install with: "
            "pip install git+https://github.com/TorchSim/torch-sim"
        )


def _load_atoms(atoms: str | Atoms) -> Atoms:
    if isinstance(atoms, str):
        return read(atoms)
    return atoms.copy()


def _get_mace_model(model: str, device: str):
    """Load MACE model for torchsim (raw model, not ASE calculator)."""
    import torch
    from mace.calculators.foundations_models import mace_mp
    from torch_sim.models.mace import MaceModel

    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU.")
        device = "cpu"

    model_map = {"mace_mp_small": "small", "mace_mp_medium": "medium"}
    model_name = model_map.get(model, model)

    raw_model = mace_mp(model=model_name, return_raw_model=True)
    return MaceModel(model=raw_model, device=device)


def _memory_padding(memory_limit_gb: float | None) -> float:
    """Compute safe memory padding."""
    if memory_limit_gb is not None:
        import torch
        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return min(memory_limit_gb / total_gb, 0.9)
    return 0.6


def torchsim_optimize(
    atoms: str | Atoms,
    model: str,
    optimizer: str,
    fmax: float,
    max_steps: int,
    device: str,
    memory_limit_gb: float | None,
) -> Atoms:
    """Run optimization using torchsim."""
    _check_torchsim()
    import torch_sim as ts

    atoms = _load_atoms(atoms)
    mace_model = _get_mace_model(model, device)

    opt_map = {
        "lbfgs": ts.Optimizer.lbfgs,
        "fire": ts.Optimizer.fire,
        "bfgs": ts.Optimizer.bfgs,
    }
    if optimizer not in opt_map:
        raise ValueError(f"Unknown optimizer '{optimizer}'. Use: {list(opt_map.keys())}")

    convergence_fn = ts.generate_force_convergence_fn(force_tol=fmax)
    padding = _memory_padding(memory_limit_gb)

    result = ts.optimize(
        system=atoms,
        model=mace_model,
        optimizer=opt_map[optimizer],
        convergence_fn=convergence_fn,
        max_steps=max_steps,
        autobatcher=True,
        max_memory_padding=padding,
    )

    return result.to_atoms()[0]


def torchsim_run_md(
    atoms: str | Atoms,
    temperature: float,
    n_steps: int,
    timestep: float,
    model: str,
    thermostat: str,
    log_interval: int,
    equilibration_steps: int,
    device: str,
    memory_limit_gb: float | None,
    trajectory: str | None,
) -> list[Atoms]:
    """Run MD using torchsim."""
    _check_torchsim()
    import torch_sim as ts

    atoms = _load_atoms(atoms)
    mace_model = _get_mace_model(model, device)

    integrator_map = {
        "langevin": ts.Integrator.nvt_langevin,
        "nose_hoover": ts.Integrator.nvt_nose_hoover,
    }
    if thermostat not in integrator_map:
        raise ValueError(f"Unknown thermostat '{thermostat}'. Use: {list(integrator_map.keys())}")

    if equilibration_steps > 0:
        ts.integrate(
            system=atoms,
            model=mace_model,
            n_steps=equilibration_steps,
            timestep=timestep / 1000,
            temperature=temperature,
            integrator=integrator_map[thermostat],
            autobatcher=True,
        )

    traj_files = [trajectory] if trajectory else None
    reporter = None
    if traj_files:
        reporter = dict(filenames=traj_files, state_frequency=log_interval)

    result = ts.integrate(
        system=atoms,
        model=mace_model,
        n_steps=n_steps,
        timestep=timestep / 1000,
        temperature=temperature,
        integrator=integrator_map[thermostat],
        trajectory_reporter=reporter,
        autobatcher=True,
    )

    if trajectory:
        traj = ts.TorchSimTrajectory(trajectory)
        snapshots = []
        for i in range(0, len(traj), log_interval):
            snapshots.append(traj[i].to_atoms()[0])
        return snapshots

    return [result.to_atoms()[0]]
