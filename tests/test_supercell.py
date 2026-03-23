import numpy as np
import pytest
from ase import Atoms

from defectmof.supercell import build_supercell


def test_build_supercell_2x2x2_atom_count(two_tiny_cells):
    """2x2x2 supercell of 4-atom cells should have ~32 atoms (minus overlaps)."""
    defective, pristine = two_tiny_cells
    sc = build_supercell(defective, pristine, size=(2, 2, 2), mode="random")
    assert isinstance(sc, Atoms)
    assert len(sc) > 0
    assert len(sc) <= 32


def test_build_supercell_defect_fraction(two_tiny_cells):
    """With 8 cells and fraction=0.5, exactly 4 should be defective."""
    defective, pristine = two_tiny_cells
    sc = build_supercell(defective, pristine, size=(2, 2, 2), defect_fraction=0.5)
    n_br = sum(1 for s in sc.get_chemical_symbols() if s == "Br")
    assert n_br >= 1


def test_build_supercell_all_modes(two_tiny_cells):
    """All arrangement modes should produce valid Atoms objects."""
    defective, pristine = two_tiny_cells
    modes = [
        "random", "alternating_a", "alternating_b", "alternating_c",
        "alternating_ab", "alternating_ac", "alternating_bc", "alternating_abc",
        "clustered_small", "clustered_large",
    ]
    for mode in modes:
        sc = build_supercell(defective, pristine, size=(2, 2, 2), mode=mode)
        assert isinstance(sc, Atoms), f"Mode {mode} failed"
        assert len(sc) > 0, f"Mode {mode} produced empty structure"


def test_build_supercell_cell_mismatch_raises():
    """Mismatched cell parameters should raise ValueError."""
    cell_a = Atoms("Al", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)
    cell_b = Atoms("Al", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    with pytest.raises(ValueError, match="cell parameters"):
        build_supercell(cell_a, cell_b, size=(2, 2, 2))


def test_build_supercell_reproducible(two_tiny_cells):
    """Same seed should produce identical results."""
    defective, pristine = two_tiny_cells
    sc1 = build_supercell(defective, pristine, size=(2, 2, 2), mode="random", seed=42)
    sc2 = build_supercell(defective, pristine, size=(2, 2, 2), mode="random", seed=42)
    np.testing.assert_array_almost_equal(sc1.positions, sc2.positions)


def test_build_supercell_from_cif_path(cif_paths):
    """Should accept string paths to CIF files."""
    sc = build_supercell(
        cif_paths["defective"], cif_paths["pristine"],
        size=(2, 2, 2), mode="random",
    )
    assert isinstance(sc, Atoms)
    assert len(sc) > 1000
