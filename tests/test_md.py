import pytest
from ase import Atoms
from unittest.mock import patch

from defectmof.md import run_md


def test_run_md_returns_list_of_atoms(tiny_atoms):
    mock_snapshots = [tiny_atoms.copy() for _ in range(5)]
    with patch("defectmof.md.ase_run_md", return_value=mock_snapshots):
        result = run_md(tiny_atoms, n_steps=50, backend="ase")
        assert isinstance(result, list)
        assert all(isinstance(a, Atoms) for a in result)


def test_run_md_invalid_backend(tiny_atoms):
    with pytest.raises(ValueError, match="backend"):
        run_md(tiny_atoms, backend="unknown")


def test_run_md_invalid_thermostat(tiny_atoms):
    with pytest.raises(ValueError, match="thermostat"):
        run_md(tiny_atoms, thermostat="berendsen")
