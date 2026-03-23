"""Example: Run MD simulation at 150K for PDF computation.

Uses the VASP-optimized pristine MOF-303 structure.
"""
from defectmof import run_md

snapshots = run_md(
    atoms="examples/structures/CONTCAR_optimized.cif",
    temperature=150.0,
    n_steps=5000,
    timestep=1.0,
    model="mace_mp_small",
    thermostat="langevin",
    log_interval=10,
    equilibration_steps=1000,
    device="cuda",
    trajectory="md_150K_pristine.traj",
)
print(f"Collected {len(snapshots)} snapshots")
