import numpy as np
import pytest
from ase import Atoms

from defectmof.pdf import compute_rdf


def test_compute_rdf_returns_arrays(tiny_atoms):
    snapshots = [tiny_atoms.copy() for _ in range(3)]
    r, g_r = compute_rdf(snapshots, rmax=5.0, nbins=50)
    assert isinstance(r, np.ndarray)
    assert isinstance(g_r, np.ndarray)
    assert len(r) == len(g_r)
    assert len(r) == 50


def test_compute_rdf_partial(tiny_atoms):
    snapshots = [tiny_atoms.copy()]
    r, g_r = compute_rdf(snapshots, rmax=5.0, nbins=50, elements=("Al", "O"))
    assert len(r) == 50


def test_compute_rdf_single_snapshot(tiny_atoms):
    r, g_r = compute_rdf([tiny_atoms], rmax=5.0, nbins=50)
    assert len(r) == 50


# --- Edge case tests ---


def test_compute_rdf_empty_snapshots():
    """Empty snapshot list should raise a clear ValueError."""
    with pytest.raises(ValueError, match="No snapshots"):
        compute_rdf([], rmax=5.0, nbins=50)


def test_compute_pdf_invalid_engine(tiny_atoms):
    """Invalid engine should raise ValueError."""
    from defectmof.pdf import compute_pdf
    with pytest.raises(ValueError, match="engine"):
        compute_pdf([tiny_atoms], engine="invalid_engine")


def test_compute_pdf_diffpy_not_installed(tiny_atoms):
    """Should raise ImportError with helpful message when diffpy missing."""
    from defectmof.pdf import compute_pdf
    import unittest.mock as mock

    with mock.patch.dict('sys.modules', {'diffpy': None, 'diffpy.srreal': None, 'diffpy.srreal.pdfcalculator': None, 'diffpy.structure': None}):
        with pytest.raises(ImportError, match="diffpy"):
            compute_pdf([tiny_atoms], engine="diffpy")


def test_compute_pdf_rmin_ge_rmax(tiny_atoms):
    from defectmof.pdf import compute_pdf
    with pytest.raises(ValueError, match="rmin"):
        compute_pdf([tiny_atoms], rmin=30.0, rmax=10.0)

def test_compute_rdf_invalid_rmax(tiny_atoms):
    with pytest.raises(ValueError, match="rmax"):
        compute_rdf([tiny_atoms], rmax=-5.0)

def test_compute_rdf_invalid_nbins(tiny_atoms):
    with pytest.raises(ValueError, match="nbins"):
        compute_rdf([tiny_atoms], nbins=0)

def test_compute_pdf_empty_snapshots(tiny_atoms):
    from defectmof.pdf import compute_pdf
    with pytest.raises(ValueError, match="No snapshots"):
        compute_pdf([], engine="diffpy")


def test_compute_rdf_very_small_cell():
    """Cell smaller than 2*rmax should still work (fallback in code)."""
    tiny = Atoms(
        symbols=["Al", "O"],
        positions=[[0, 0, 0], [1.0, 0, 0]],
        cell=[2.0, 2.0, 2.0],
        pbc=True,
    )
    # rmax=5.0 > cell/2=1.0 — this tests the small cell handling
    r, g_r = compute_rdf([tiny], rmax=5.0, nbins=50)
    assert len(r) == 50
