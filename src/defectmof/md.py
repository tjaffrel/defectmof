"""Molecular dynamics simulation using MACE force fields."""

from ase import Atoms

from defectmof._ase import ase_run_md


def run_md(
    atoms: str | Atoms,
    temperature: float = 150.0,
    n_steps: int = 5000,
    timestep: float = 1.0,
    model: str = "mace_mp_medium",
    thermostat: str = "langevin",
    friction: float = 0.01,
    log_interval: int = 10,
    equilibration_steps: int = 1000,
    backend: str = "ase",
    device: str = "cuda",
    memory_limit_gb: float | None = None,
    trajectory: str | None = None,
    head: str | None = None,
) -> list[Atoms]:
    """Run molecular dynamics simulation.

    Args:
        atoms: Path to CIF or ASE Atoms.
        temperature: Simulation temperature in Kelvin.
        n_steps: Number of production MD steps.
        timestep: Timestep in femtoseconds.
        model: MACE model name or path to .model file.
        thermostat: "langevin" or "nose_hoover".
        friction: Langevin friction coefficient (1/fs). Ignored for nose_hoover.
        log_interval: Save a snapshot every N steps.
        equilibration_steps: Discard first N steps before collecting snapshots.
        backend: "ase" or "torchsim".
        device: "cuda" or "cpu".
        memory_limit_gb: Max GPU memory for torchsim.
        trajectory: Path to save trajectory file.
        head: For multi-head MACE models, which head to use (e.g. "pt_head").

    Returns:
        List of ASE Atoms snapshots from the production phase.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive (Kelvin), got {temperature}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}")
    if timestep <= 0:
        raise ValueError(f"timestep must be positive (fs), got {timestep}")
    if log_interval < 1:
        raise ValueError(f"log_interval must be >= 1, got {log_interval}")
    if equilibration_steps < 0:
        raise ValueError(f"equilibration_steps must be >= 0, got {equilibration_steps}")

    if thermostat not in ("langevin", "nose_hoover"):
        raise ValueError(f"Unknown thermostat '{thermostat}'. Use: 'langevin' or 'nose_hoover'")

    if backend == "ase":
        return ase_run_md(
            atoms, temperature, n_steps, timestep, model, thermostat,
            friction, log_interval, equilibration_steps, device, trajectory,
            head=head,
        )
    elif backend == "torchsim":
        from defectmof._torchsim import torchsim_run_md
        return torchsim_run_md(
            atoms, temperature, n_steps, timestep, model, thermostat,
            log_interval, equilibration_steps, device, memory_limit_gb, trajectory,
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use: 'ase' or 'torchsim'")
