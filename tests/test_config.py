import pytest
from bysopt.config import QuantumWellConfig, parse_args


def test_config_initialization(config):
    assert len(config.PARAM_NAMES) == 9
    assert config.TARGET_WAVELENGTH == 4.0
    assert config.WPE_WEIGHT == 0.7
    assert config.WAVELENGTH_WEIGHT == 0.3
    assert config.STRAIN_TOLERANCE == 0.1

def test_parse_args():
    args = parse_args([])
    assert args.max_iterations == 20
    assert args.data_file == '../result/quantum_well_opt.csv'
    assert args.batch_size == 3

def test_parse_args_custom():
    test_args = [
        '--max_iterations', '30',
        '--target_wavelength', '4.2',
        '--batch_size', '5'
    ]
    args = parse_args(test_args)
    assert args.max_iterations == 30
    assert args.target_wavelength == 4.2
    assert args.batch_size == 5