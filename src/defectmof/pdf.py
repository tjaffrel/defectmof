"""Pair Distribution Function and Radial Distribution Function computation."""

import os
import tempfile
import warnings

import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory


def _load_snapshots(snapshots: list[Atoms] | str) -> list[Atoms]:
    """Load snapshots from trajectory path or return as-is."""
    if isinstance(snapshots, str):
        traj = Trajectory(snapshots)
        return [atoms for atoms in traj]
    return snapshots


def compute_pdf(
    snapshots: list[Atoms] | str,
    rmin: float = 1.0,
    rmax: float = 30.0,
    rstep: float = 0.01,
    scattering: str = "xray",
    engine: str = "diffpy",
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Pair Distribution Function G(r) from MD snapshots.

    Args:
        snapshots: List of ASE Atoms or path to trajectory file.
        rmin: Minimum r in Angstroms (values below 1.0 are rarely physical).
        rmax: Maximum r in Angstroms.
        rstep: Step size for r grid.
        scattering: "xray" or "neutron" (diffpy engine only).
        engine: "diffpy" for diffpy-CMI or "debye_calculator" for GPU-accelerated.
        device: Compute device for debye_calculator.

    Returns:
        (r, G_r): Distances and frame-averaged PDF.
    """
    frames = _load_snapshots(snapshots)
    if len(frames) == 0:
        raise ValueError("No snapshots provided. Pass at least one Atoms object.")

    if rmin >= rmax:
        raise ValueError(f"rmin ({rmin}) must be less than rmax ({rmax})")
    if rstep <= 0:
        raise ValueError(f"rstep must be positive, got {rstep}")
    if scattering not in ("xray", "neutron"):
        raise ValueError(f"scattering must be 'xray' or 'neutron', got '{scattering}'")

    if engine == "diffpy":
        return _pdf_diffpy(frames, rmin, rmax, rstep, scattering)
    elif engine == "debye_calculator":
        return _pdf_debye(frames, rmin, rmax, rstep, device)
    else:
        raise ValueError(f"Unknown engine '{engine}'. Use: 'diffpy' or 'debye_calculator'")


def _pdf_diffpy(
    frames: list[Atoms], rmin: float, rmax: float, rstep: float, scattering: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PDF using diffpy-CMI."""
    try:
        from diffpy.srreal.pdfcalculator import PDFCalculator
        from diffpy.structure import loadStructure
    except ImportError:
        raise ImportError(
            "diffpy is not installed. Install with: pip install diffpy.srreal"
        )

    calc = PDFCalculator()
    calc.rmin = rmin
    calc.rmax = rmax
    calc.rstep = rstep
    calc.setScatteringType("X" if scattering == "xray" else "N")

    all_gr = []
    for atoms in frames:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
                tmp_path = f.name
            atoms.write(tmp_path, format="cif")
            stru = loadStructure(tmp_path)
            r, gr = calc(stru)
            all_gr.append(gr)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    r = np.array(r)
    g_r = np.mean(all_gr, axis=0)
    return r, g_r


def _pdf_debye(
    frames: list[Atoms], rmin: float, rmax: float, rstep: float, device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PDF using DebyeCalculator (GPU-accelerated)."""
    try:
        from debyecalculator import DebyeCalculator
    except ImportError:
        raise ImportError(
            "DebyeCalculator is not installed. Install with: pip install debyecalculator"
        )

    dc = DebyeCalculator(device=device, rmin=rmin, rmax=rmax, rstep=rstep)

    all_gr = []
    for atoms in frames:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
                tmp_path = f.name
            atoms.write(tmp_path, format="xyz")
            r, gr = dc.gr(tmp_path)
            all_gr.append(gr)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    r = np.array(r)
    g_r = np.mean(all_gr, axis=0)
    return r, g_r


def compute_rdf(
    snapshots: list[Atoms] | str,
    rmax: float = 10.0,
    nbins: int = 200,
    elements: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the radial distribution function g(r) using ASE.

    This is a quick, unweighted RDF — not directly comparable to
    experimental PDF. Use compute_pdf() for experimental comparison.

    Args:
        snapshots: List of ASE Atoms or path to trajectory file.
        rmax: Maximum distance in Angstroms.
        nbins: Number of bins.
        elements: Tuple of two element symbols for partial RDF, e.g. ("Al", "O").

    Returns:
        (r, g_r): Distances and frame-averaged g(r).
    """
    from ase.geometry.rdf import get_rdf

    frames = _load_snapshots(snapshots)
    if len(frames) == 0:
        raise ValueError("No snapshots provided. Pass at least one Atoms object.")

    if rmax <= 0:
        raise ValueError(f"rmax must be positive, got {rmax}")
    if nbins < 1:
        raise ValueError(f"nbins must be >= 1, got {nbins}")

    all_rdf = []
    for atoms in frames:
        try:
            rdf, rr = get_rdf(atoms, rmax, nbins, elements=elements, no_dists=False)
        except Exception:
            # Cell too small for periodic RDF — fall back to non-periodic
            atoms_np = atoms.copy()
            atoms_np.set_pbc(False)
            rdf, rr = get_rdf(atoms_np, rmax, nbins, elements=elements, no_dists=False)
        all_rdf.append(rdf)

    g_r = np.mean(all_rdf, axis=0)
    dr = rmax / nbins
    r = np.linspace(dr / 2, rmax - dr / 2, nbins)
    return r, g_r
