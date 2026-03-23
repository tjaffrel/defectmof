import os

import numpy as np
from ase import Atoms

from defectmof.utils import fix_short_cbr_bonds, prune_overlapping_atoms


def test_fix_short_cbr_bonds_moves_br():
    """A Br atom very close to C should be moved to target_dist."""
    atoms = Atoms(
        symbols=["C", "Br"],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]],
        cell=[10, 10, 10],
        pbc=True,
    )
    fixed = fix_short_cbr_bonds(atoms, target_dist=2.06)
    dist = np.linalg.norm(fixed.positions[1] - fixed.positions[0])
    assert abs(dist - 2.06) < 0.01


def test_fix_short_cbr_bonds_ignores_normal_bonds():
    """Bonds longer than 1.0A should not be touched."""
    atoms = Atoms(
        symbols=["C", "Br"],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
        cell=[10, 10, 10],
        pbc=True,
    )
    original_pos = atoms.positions.copy()
    fix_short_cbr_bonds(atoms, target_dist=2.06)
    np.testing.assert_array_almost_equal(atoms.positions, original_pos)


def test_prune_overlapping_atoms_removes_duplicates():
    """Two atoms at nearly the same position — one should be removed."""
    atoms = Atoms(
        symbols=["Al", "Al", "O"],
        positions=[[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [3.0, 3.0, 3.0]],
        cell=[10, 10, 10],
        pbc=True,
    )
    pruned = prune_overlapping_atoms(atoms, threshold=0.8)
    assert len(pruned) == 2


def test_prune_overlapping_atoms_keeps_distant():
    """All atoms far apart — none should be removed."""
    atoms = Atoms(
        symbols=["Al", "O", "C"],
        positions=[[0, 0, 0], [3, 0, 0], [0, 3, 0]],
        cell=[10, 10, 10],
        pbc=True,
    )
    pruned = prune_overlapping_atoms(atoms, threshold=0.8)
    assert len(pruned) == 3


# --- Edge case tests ---


def test_fix_short_cbr_bonds_no_br_atoms():
    """Structure with no Br should be returned unchanged."""
    atoms = Atoms(
        symbols=["C", "N", "O"],
        positions=[[0, 0, 0], [0.5, 0, 0], [1, 0, 0]],
        cell=[10, 10, 10],
        pbc=True,
    )
    original = atoms.positions.copy()
    fix_short_cbr_bonds(atoms)
    np.testing.assert_array_equal(atoms.positions, original)


def test_fix_short_cbr_bonds_multiple_pairs():
    """Multiple C-Br pairs should all be fixed."""
    atoms = Atoms(
        symbols=["C", "Br", "C", "Br"],
        positions=[[0, 0, 0], [0, 0, 0.3], [5, 5, 5], [5, 5, 5.4]],
        cell=[10, 10, 10],
        pbc=True,
    )
    fixed = fix_short_cbr_bonds(atoms, target_dist=2.06)
    # Both Br atoms should have been moved
    dist1 = np.linalg.norm(fixed.positions[1] - fixed.positions[0])
    dist2 = np.linalg.norm(fixed.positions[3] - fixed.positions[2])
    assert abs(dist1 - 2.06) < 0.05
    assert abs(dist2 - 2.06) < 0.05


def test_prune_overlapping_atoms_preserves_species():
    """After pruning, remaining atoms should have correct species."""
    atoms = Atoms(
        symbols=["Al", "O", "O"],
        positions=[[0, 0, 0], [5, 5, 5], [5.01, 5, 5]],
        cell=[20, 20, 20],
        pbc=True,
    )
    pruned = prune_overlapping_atoms(atoms, threshold=0.8)
    assert len(pruned) == 2
    symbols = pruned.get_chemical_symbols()
    assert "Al" in symbols  # Al should survive (not overlapping)


def test_prune_overlapping_atoms_single_atom():
    """Single atom should remain untouched."""
    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    pruned = prune_overlapping_atoms(atoms, threshold=0.8)
    assert len(pruned) == 1


def test_prune_overlapping_atoms_chain_of_close_atoms():
    """A chain of atoms all within threshold — should remove most."""
    atoms = Atoms(
        symbols=["Al"] * 5,
        positions=[[0, 0, i * 0.3] for i in range(5)],  # all within 0.3A of neighbors
        cell=[10, 10, 10],
        pbc=True,
    )
    pruned = prune_overlapping_atoms(atoms, threshold=0.8)
    # Should keep at least 1 but remove most
    assert len(pruned) < 5
    assert len(pruned) >= 1


def test_visualize_distribution_creates_file(tmp_path):
    """Should create a PNG file at the specified path."""
    from defectmof.utils import visualize_distribution
    grid = np.array([[[1, 2], [2, 1]], [[1, 1], [2, 2]]])
    output = str(tmp_path / "test_dist.png")
    visualize_distribution(grid, "test", output_path=output)
    assert os.path.exists(output)


def test_visualize_distribution_default_path(tmp_path, monkeypatch):
    """Without output_path, should create distribution_{tag}.png."""
    from defectmof.utils import visualize_distribution
    monkeypatch.chdir(tmp_path)
    grid = np.array([[[1, 2], [2, 1]], [[1, 1], [2, 2]]])
    visualize_distribution(grid, "mytest")
    assert os.path.exists(tmp_path / "distribution_mytest.png")
