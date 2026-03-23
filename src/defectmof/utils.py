"""Utility functions preserved from the original MOF defect code.

These function names match the original author's code for traceability.
"""

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list


def fix_short_cbr_bonds(atoms: Atoms, target_dist: float = 2.06) -> Atoms:
    """Move Br atoms along Z to reach target_dist if C-Br bond is < 1.0 A.

    Args:
        atoms: ASE Atoms object containing C and Br atoms.
        target_dist: Target C-Br bond distance in Angstroms.

    Returns:
        The same Atoms object with corrected Br positions.
    """
    i_list, j_list, dists = neighbor_list("ijd", atoms, 1.0)
    for i, j, d in zip(i_list, j_list, dists):
        symbols = [atoms[i].symbol, atoms[j].symbol]
        if "C" in symbols and "Br" in symbols:
            br_idx = i if atoms[i].symbol == "Br" else j
            c_idx = j if atoms[i].symbol == "Br" else i
            c_pos = atoms[c_idx].position
            br_pos = atoms[br_idx].position

            z_direction = 1 if br_pos[2] >= c_pos[2] else -1
            dx, dy = br_pos[0] - c_pos[0], br_pos[1] - c_pos[1]
            h_sq = dx**2 + dy**2
            v_diff_sq = target_dist**2 - h_sq

            if v_diff_sq > 0:
                new_dz = np.sqrt(v_diff_sq) * z_direction
                atoms.positions[br_idx, 2] = c_pos[2] + new_dz
    return atoms


def prune_overlapping_atoms(atoms: Atoms, threshold: float = 0.8) -> Atoms:
    """Remove overlapping atoms within threshold distance.

    Args:
        atoms: ASE Atoms object.
        threshold: Distance in Angstroms below which atoms are considered overlapping.

    Returns:
        The same Atoms object with overlapping atoms removed.
    """
    i_arr, j_arr = neighbor_list("ij", atoms, threshold)
    to_remove = set(j_arr[i_arr < j_arr])
    if to_remove:
        del atoms[[idx for idx in sorted(to_remove, reverse=True)]]
    return atoms


def visualize_distribution(
    grid_labels: np.ndarray, tag: str, output_path: str | None = None
) -> None:
    """Save a 3D scatter plot of defective (blue) vs pristine (red) sites.

    Args:
        grid_labels: 3D numpy array where 1=defective, 2=pristine.
        tag: Label for the plot title and default filename.
        output_path: Where to save the plot. Defaults to 'distribution_{tag}.png'.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    nx, ny, nz = grid_labels.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                color = "blue" if grid_labels[i, j, k] == 1 else "red"
                ax.scatter(i, j, k, c=color, s=100, edgecolors="k")
    n_defective = int(np.sum(grid_labels == 1))
    n_pristine = int(np.sum(grid_labels == 2))
    ax.set_title(f"Distribution: {tag} ({n_defective} defective, {n_pristine} pristine)")
    path = output_path or f"distribution_{tag}.png"
    plt.savefig(path)
    plt.close()
