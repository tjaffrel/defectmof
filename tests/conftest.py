import pytest
import numpy as np
from ase import Atoms


@pytest.fixture
def tiny_atoms():
    """A minimal 8-atom cubic structure for fast tests (no GPU needed)."""
    atoms = Atoms(
        symbols=["Al", "O", "C", "N", "H", "Al", "O", "C"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [1.5, 1.5, 0.0],
            [0.0, 0.0, 1.5],
            [1.5, 0.0, 1.5],
            [0.0, 1.5, 1.5],
            [1.5, 1.5, 1.5],
        ],
        cell=[3.0, 3.0, 3.0],
        pbc=True,
    )
    return atoms


@pytest.fixture
def two_tiny_cells():
    """Two tiny unit cells for supercell building tests."""
    cell = [3.0, 3.0, 3.0]
    defective = Atoms(
        symbols=["Al", "O", "Br", "C"],
        positions=[[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [1.5, 1.5, 0]],
        cell=cell,
        pbc=True,
    )
    pristine = Atoms(
        symbols=["Al", "O", "N", "C"],
        positions=[[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [1.5, 1.5, 0]],
        cell=cell,
        pbc=True,
    )
    return defective, pristine


@pytest.fixture
def cif_paths():
    """Paths to the real CIF structures in examples/."""
    from pathlib import Path

    base = Path(__file__).parent.parent / "examples" / "structures"
    return {
        "defective": str(base / "struct1_optimized_c_axis.cif"),
        "pristine": str(base / "struct2_optimized_a_axis.cif"),
        "contcar_final": str(base / "CONTCAR_final.cif"),
        "contcar_optimized": str(base / "CONTCAR_optimized.cif"),
    }
