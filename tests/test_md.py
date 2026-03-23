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


# --- Edge case tests ---


def test_run_md_zero_steps(tiny_atoms):
    """n_steps=0 should return empty list."""
    from unittest.mock import call
    with patch("defectmof.md.ase_run_md", return_value=[]) as mock_md:
        result = run_md(tiny_atoms, n_steps=0, equilibration_steps=0, backend="ase")
        assert isinstance(result, list)
        assert len(result) == 0
        # Verify n_steps=0 and equilibration_steps=0 were forwarded
        args = mock_md.call_args
        assert args[0][2] == 0  # n_steps


def test_run_md_invalid_temperature(tiny_atoms):
    with pytest.raises(ValueError, match="temperature"):
        run_md(tiny_atoms, temperature=-10)

def test_run_md_invalid_timestep(tiny_atoms):
    with pytest.raises(ValueError, match="timestep"):
        run_md(tiny_atoms, timestep=0)

def test_run_md_invalid_log_interval(tiny_atoms):
    with pytest.raises(ValueError, match="log_interval"):
        run_md(tiny_atoms, log_interval=0)
