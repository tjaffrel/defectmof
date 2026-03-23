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
