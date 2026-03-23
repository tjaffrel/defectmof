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
