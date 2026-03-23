"""Tests that Hydra configs parse correctly."""
from omegaconf import OmegaConf


def test_optimize_config_loads():
    cfg = OmegaConf.load("examples/configs/optimize.yaml")
    assert cfg.model == "mace_mp_medium"
    assert cfg.optimizer == "lbfgs"
    assert cfg.fmax == 0.05


def test_supercell_config_loads():
    cfg = OmegaConf.load("examples/configs/supercell.yaml")
    assert cfg.defect_fraction == 0.44
    assert list(cfg.size) == [4, 4, 4]


def test_md_config_loads():
    cfg = OmegaConf.load("examples/configs/md_150K.yaml")
    assert cfg.temperature == 150.0
    assert cfg.thermostat == "langevin"


def test_pdf_config_loads():
    cfg = OmegaConf.load("examples/configs/pdf_from_md.yaml")
    assert cfg.scattering == "xray"
    assert cfg.engine == "diffpy"
