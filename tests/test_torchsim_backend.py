"""Tests for torchsim backend (mocked — no actual torchsim needed)."""

import pytest
from unittest.mock import patch

from defectmof._torchsim import _check_torchsim, _memory_padding


def test_check_torchsim_not_installed():
    """Should raise ImportError with install instructions."""
    with patch.dict("sys.modules", {"torch_sim": None}):
        with pytest.raises(ImportError, match="torchsim is not installed"):
            _check_torchsim()


def test_memory_padding_default():
    """Without memory_limit_gb, should return 0.6."""
    assert _memory_padding(None) == 0.6


def test_memory_padding_with_limit():
    """With memory_limit_gb, should compute proportional padding."""
    # Mock a GPU with 24GB
    mock_props = type("Props", (), {"total_memory": 24 * 1024**3})()
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            # 16GB limit on 24GB GPU = 0.667 padding
            padding = _memory_padding(16.0)
            assert 0.6 < padding < 0.7

            # 30GB limit on 24GB GPU should cap at 0.9
            padding = _memory_padding(30.0)
            assert padding == 0.9


def test_memory_padding_no_cuda():
    """Without CUDA, should return default 0.6 even with limit set."""
    with patch("torch.cuda.is_available", return_value=False):
        assert _memory_padding(16.0) == 0.6
