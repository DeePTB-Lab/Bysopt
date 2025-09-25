import pytest
import torch
import numpy as np
from bysopt.constraints import PhysicalConstraintProcessor


def test_strain_weight_vector(config):
    processor = PhysicalConstraintProcessor(config)
    assert len(processor.weight_epsilon) == 8
    assert processor.weight_epsilon[0] == config.EPSILON_B
    assert processor.weight_epsilon[1] == config.EPSILON_W


def test_apply_constraints_valid(config):
    processor = PhysicalConstraintProcessor(config)

    params = torch.tensor([
        [3.0, 1.5, 1.2, 4.0, 1.5, 3.0, 2.5, 2.5, 140],  # valid
        [3.5, 1.6, 1.3, 4.5, 1.6, 3.5, 2.6, 2.6, 150]  # valid
    ], dtype=torch.float32)

    result, strained_tol = processor.apply_constraints(params)

    assert len(result) == 2
    assert len(strained_tol) == 2
    assert all(np.abs(strain) < config.STRAIN_TOLERANCE for strain in strained_tol)


def test_apply_constraints_invalid(config):
    processor = PhysicalConstraintProcessor(config)

    params = torch.tensor([
        [4.75, 1.2, 1.65, 3.65, 2.0, 2.55, 3.35, 2.05, 110],  # invalid
        [4.75, 1.2, 1.65, 3.65, 2.0, 2.55, 3.35, 2.05, 110]  # invalid
    ], dtype=torch.float32)

    result, strained_tol = processor.apply_constraints(params)

    assert result is not None
    assert strained_tol is not None


def test_apply_constraints_mixed(config):
    processor = PhysicalConstraintProcessor(config)

    params = torch.tensor([
        [3.0, 1.5, 1.2, 4.0, 1.5, 3.0, 2.5, 2.5, 140],  # valid
        [4.75, 1.2, 1.65, 3.65, 2.0, 2.55, 3.35, 2.05, 110]  # invalid
    ], dtype=torch.float32)

    result, strained_tol = processor.apply_constraints(params)

    assert len(result) >= 1