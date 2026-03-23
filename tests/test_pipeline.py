import pytest
from ase import Atoms
from unittest.mock import patch

from defectmof.pipeline import hierarchical_optimize


def test_hierarchical_optimize_validates_size():
    cell_a = Atoms("Al", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)
    cell_b = Atoms("Al", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)

    with pytest.raises(ValueError, match="must be >= 4"):
        hierarchical_optimize(cell_a, cell_b, target_size=(2, 2, 2))

    with pytest.raises(ValueError, match="divisible by 2"):
        hierarchical_optimize(cell_a, cell_b, target_size=(5, 5, 5))


def test_hierarchical_optimize_calls_stages(two_tiny_cells):
    defective, pristine = two_tiny_cells
    call_log = []

    def mock_build(*args, **kwargs):
        call_log.append(("build", kwargs.get("size")))
        return defective.copy()

    def mock_optimize(atoms, **kwargs):
        call_log.append(("optimize", kwargs.get("fmax")))
        return atoms

    with patch("defectmof.pipeline.build_supercell", side_effect=mock_build):
        with patch("defectmof.pipeline.optimize", side_effect=mock_optimize):
            hierarchical_optimize(defective, pristine, target_size=(4, 4, 4))

    build_calls = [c for c in call_log if c[0] == "build"]
    assert len(build_calls) >= 1
