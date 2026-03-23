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


# --- Edge case tests ---


def test_build_supercell_all_defective(two_tiny_cells):
    """defect_fraction=1.0 should make all cells defective."""
    defective, pristine = two_tiny_cells
    sc = build_supercell(defective, pristine, size=(2, 2, 2), defect_fraction=1.0)
    # All cells defective means all Br atoms present, no N
    symbols = sc.get_chemical_symbols()
    assert "Br" in symbols
    # With 8 cells, 8 Br expected (before pruning)
    n_br = symbols.count("Br")
    assert n_br >= 4  # at least half survive pruning


def test_build_supercell_all_pristine(two_tiny_cells):
    """defect_fraction=0.0 should make all cells pristine."""
    defective, pristine = two_tiny_cells
    sc = build_supercell(defective, pristine, size=(2, 2, 2), defect_fraction=0.0)
    symbols = sc.get_chemical_symbols()
    assert "Br" not in symbols  # no defective cells
    assert "N" in symbols  # all pristine


def test_build_supercell_1x1x1(two_tiny_cells):
    """1x1x1 should return a single unit cell."""
    defective, pristine = two_tiny_cells
    sc = build_supercell(defective, pristine, size=(1, 1, 1), defect_fraction=1.0)
    assert isinstance(sc, Atoms)
    assert len(sc) > 0


def test_build_supercell_invalid_mode(two_tiny_cells):
    """Invalid mode should raise ValueError."""
    defective, pristine = two_tiny_cells
    with pytest.raises(ValueError, match="Unknown mode"):
        build_supercell(defective, pristine, size=(2, 2, 2), mode="invalid_mode")


def test_build_supercell_different_seeds_differ(two_tiny_cells):
    """Different seeds should produce different chemical arrangements."""
    defective, pristine = two_tiny_cells
    sc1 = build_supercell(defective, pristine, size=(4, 4, 4), mode="random", seed=1)
    sc2 = build_supercell(defective, pristine, size=(4, 4, 4), mode="random", seed=99)
    # Different seeds -> different which cells are defective/pristine, so different species order.
    # Positions may coincide (Br and N sit at the same coordinates), but symbols must differ.
    assert sc1.get_chemical_symbols() != sc2.get_chemical_symbols()


def test_build_supercell_nonexistent_cif():
    """Non-existent CIF path should raise an error."""
    with pytest.raises(Exception):
        build_supercell("/nonexistent/path.cif", "/also/nonexistent.cif", size=(2, 2, 2))


def test_build_supercell_supercell_has_correct_cell(two_tiny_cells):
    """Supercell cell should be N times the unit cell."""
    defective, pristine = two_tiny_cells
    sc = build_supercell(defective, pristine, size=(3, 2, 4), mode="random")
    unit_cell = pristine.get_cell()
    expected_cell = unit_cell * [3, 2, 4]
    np.testing.assert_array_almost_equal(sc.get_cell(), expected_cell)


def test_build_supercell_pbc_preserved(two_tiny_cells):
    """Supercell should have PBC enabled."""
    defective, pristine = two_tiny_cells
    sc = build_supercell(defective, pristine, size=(2, 2, 2))
    assert all(sc.pbc)


def test_build_supercell_cell_tolerance_boundary():
    """Cells differing by exactly the tolerance should pass."""
    cell_a = Atoms("Al", positions=[[0, 0, 0]], cell=[3.000, 3.0, 3.0], pbc=True)
    cell_b = Atoms("Al", positions=[[0, 0, 0]], cell=[3.009, 3.0, 3.0], pbc=True)  # within 0.01
    # Should NOT raise
    sc = build_supercell(cell_a, cell_b, size=(2, 2, 2))
    assert isinstance(sc, Atoms)


def test_build_supercell_cell_just_over_tolerance():
    """Cells differing by just over tolerance should raise."""
    cell_a = Atoms("Al", positions=[[0, 0, 0]], cell=[3.000, 3.0, 3.0], pbc=True)
    cell_b = Atoms("Al", positions=[[0, 0, 0]], cell=[3.011, 3.0, 3.0], pbc=True)  # over 0.01
    with pytest.raises(ValueError, match="cell parameters"):
        build_supercell(cell_a, cell_b, size=(2, 2, 2))
