"""Tests for ASE backend error handling and edge cases."""

import pytest
from unittest.mock import patch, MagicMock
from ase import Atoms

from defectmof._ase import _load_atoms, _get_mace_calculator, ase_optimize


def test_load_atoms_from_cif(cif_paths):
    """Should load Atoms from a valid CIF path."""
    atoms = _load_atoms(cif_paths["pristine"])
    assert isinstance(atoms, Atoms)
    assert len(atoms) > 0


def test_load_atoms_nonexistent_file():
    """Should raise FileNotFoundError with clear message."""
    with pytest.raises(FileNotFoundError, match="Cannot read file"):
        _load_atoms("/nonexistent/structure.cif")


def test_load_atoms_from_atoms_object():
    """Should return a copy, not the original."""
    original = Atoms("Al", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)
    loaded = _load_atoms(original)
    assert isinstance(loaded, Atoms)
    # Should be a copy
    loaded.positions[0, 0] = 999
    assert original.positions[0, 0] != 999


def test_get_mace_calculator_cuda_fallback():
    """When CUDA unavailable, should fall back to CPU with warning."""
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.warns(match="CUDA not available"):
            # This will still try to load the model, which may fail
            # We just verify the fallback logic triggers
            try:
                _get_mace_calculator("mace_mp_small", "cuda")
            except Exception:
                pass  # Model download might fail in CI, that's ok


def test_get_mace_calculator_connection_error():
    """Should raise ConnectionError with helpful message when download fails."""
    with patch("mace.calculators.mace_mp", side_effect=Exception("urlopen error: connection refused")):
        with pytest.raises(ConnectionError, match="Cannot download MACE model"):
            _get_mace_calculator("mace_mp_small", "cpu")


def test_ase_optimize_oom_handling():
    """Should catch CUDA OOM and raise RuntimeError with helpful message."""
    import torch
    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)

    mock_calc = MagicMock()
    # Simulate OOM when forces are computed during optimization
    mock_calc.get_forces.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")

    with patch("defectmof._ase._get_mace_calculator", return_value=mock_calc):
        with patch("defectmof._ase._load_atoms", return_value=atoms):
            with pytest.raises(RuntimeError, match="GPU out of memory"):
                ase_optimize(atoms, "mace_mp_small", "lbfgs", 0.05, 10, "cuda")
