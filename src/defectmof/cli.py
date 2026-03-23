"""Hydra CLI entry points for defectmof.

Usage:
    defectmof-optimize --config-dir=examples/configs --config-name=optimize
    defectmof-build --config-dir=examples/configs --config-name=supercell
"""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=".")
def build(cfg: DictConfig) -> None:
    """Build a mixed defective/pristine supercell."""
    from defectmof.supercell import build_supercell
    from ase.io import write

    sc = build_supercell(
        defective=cfg.defective,
        pristine=cfg.pristine,
        size=tuple(cfg.size),
        defect_fraction=cfg.get("defect_fraction", 0.44),
        mode=cfg.get("mode", "random"),
        seed=cfg.get("seed", 42),
    )
    output = cfg.get("output", "supercell.cif")
    write(output, sc)
    print(f"Supercell saved to {output} ({len(sc)} atoms)")


@hydra.main(version_base=None, config_path=".")
def optimize(cfg: DictConfig) -> None:
    """Optimize a structure."""
    from defectmof.optimize import optimize as opt

    result = opt(
        atoms=cfg.input,
        model=cfg.get("model", "mace_mp_medium"),
        optimizer=cfg.get("optimizer", "lbfgs"),
        fmax=cfg.get("fmax", 0.05),
        max_steps=cfg.get("max_steps", 500),
        backend=cfg.get("backend", "ase"),
        device=cfg.get("device", "cuda"),
        output=cfg.get("output"),
    )
    print(f"Optimization complete ({len(result)} atoms)")


@hydra.main(version_base=None, config_path=".")
def md(cfg: DictConfig) -> None:
    """Run MD simulation."""
    from defectmof.md import run_md

    snapshots = run_md(
        atoms=cfg.input,
        temperature=cfg.get("temperature", 150.0),
        n_steps=cfg.get("n_steps", 5000),
        timestep=cfg.get("timestep", 1.0),
        model=cfg.get("model", "mace_mp_medium"),
        thermostat=cfg.get("thermostat", "langevin"),
        friction=cfg.get("friction", 0.01),
        log_interval=cfg.get("log_interval", 10),
        equilibration_steps=cfg.get("equilibration_steps", 1000),
        backend=cfg.get("backend", "ase"),
        device=cfg.get("device", "cuda"),
        trajectory=cfg.get("trajectory"),
    )
    print(f"MD complete: {len(snapshots)} snapshots collected")


@hydra.main(version_base=None, config_path=".")
def pdf(cfg: DictConfig) -> None:
    """Compute PDF from trajectory."""
    import numpy as np
    from defectmof.pdf import compute_pdf

    r, g_r = compute_pdf(
        snapshots=cfg.trajectory,
        rmin=cfg.get("rmin", 1.0),
        rmax=cfg.get("rmax", 30.0),
        rstep=cfg.get("rstep", 0.01),
        scattering=cfg.get("scattering", "xray"),
        engine=cfg.get("engine", "diffpy"),
        device=cfg.get("device", "cuda"),
    )
    output = cfg.get("output", "pdf_results.csv")
    np.savetxt(output, np.column_stack([r, g_r]), header="r(A)  G(r)", delimiter="  ")
    print(f"PDF saved to {output}")


@hydra.main(version_base=None, config_path=".")
def rdf(cfg: DictConfig) -> None:
    """Compute RDF from trajectory."""
    import numpy as np
    from defectmof.pdf import compute_rdf

    elements = tuple(cfg.elements) if cfg.get("elements") else None
    r, g_r = compute_rdf(
        snapshots=cfg.trajectory,
        rmax=cfg.get("rmax", 10.0),
        nbins=cfg.get("nbins", 200),
        elements=elements,
    )
    output = cfg.get("output", "rdf_results.csv")
    np.savetxt(output, np.column_stack([r, g_r]), header="r(A)  g(r)", delimiter="  ")
    print(f"RDF saved to {output}")


@hydra.main(version_base=None, config_path=".")
def hierarchical(cfg: DictConfig) -> None:
    """Run hierarchical optimization."""
    from defectmof.pipeline import hierarchical_optimize

    result = hierarchical_optimize(
        defective=cfg.defective,
        pristine=cfg.pristine,
        target_size=tuple(cfg.get("target_size", [8, 8, 8])),
        defect_fraction=cfg.get("defect_fraction", 0.44),
        mode=cfg.get("mode", "random"),
        model=cfg.get("model", "mace_mp_medium"),
        backend=cfg.get("backend", "ase"),
        device=cfg.get("device", "cuda"),
        output=cfg.get("output"),
    )
    print(f"Hierarchical optimization complete ({len(result)} atoms)")
