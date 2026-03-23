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


# --- Edge case tests ---


def test_optimize_accepts_cif_path(cif_paths):
    """optimize() should accept a CIF file path string."""
    from ase.io import read
    mock_result = read(cif_paths["pristine"])
    with patch("defectmof.optimize.ase_optimize", return_value=mock_result):
        result = optimize(cif_paths["pristine"], backend="ase")
        assert isinstance(result, Atoms)


def test_optimize_output_creates_parent_dirs_or_fails(tiny_atoms, tmp_path):
    """Output to nonexistent directory should raise."""
    bad_path = str(tmp_path / "nonexistent_dir" / "out.cif")
    mock_result = tiny_atoms.copy()
    with patch("defectmof.optimize.ase_optimize", return_value=mock_result):
        with pytest.raises(Exception):
            optimize(tiny_atoms, backend="ase", output=bad_path)
