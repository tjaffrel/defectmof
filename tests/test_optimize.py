import pytest
from ase import Atoms
from unittest.mock import patch

from defectmof.optimize import optimize


def test_optimize_returns_atoms(tiny_atoms):
    mock_result = tiny_atoms.copy()
    with patch("defectmof.optimize.ase_optimize", return_value=mock_result):
        result = optimize(tiny_atoms, backend="ase")
        assert isinstance(result, Atoms)


def test_optimize_invalid_backend(tiny_atoms):
    with pytest.raises(ValueError, match="backend"):
        optimize(tiny_atoms, backend="unknown")


def test_optimize_invalid_optimizer(tiny_atoms):
    with pytest.raises(ValueError, match="optimizer"):
        optimize(tiny_atoms, optimizer="adam")


def test_optimize_writes_output(tiny_atoms, tmp_path):
    output_path = str(tmp_path / "out.cif")
    mock_result = tiny_atoms.copy()
    with patch("defectmof.optimize.ase_optimize", return_value=mock_result):
        result = optimize(tiny_atoms, backend="ase", output=output_path)
        assert isinstance(result, Atoms)
        assert (tmp_path / "out.cif").exists()
