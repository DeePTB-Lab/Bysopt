import numpy as np
import torch
from typing import Tuple
from bysopt.config import QuantumWellConfig


class PhysicalConstraintProcessor:
    def __init__(self, config: QuantumWellConfig):
        self.config = config
        self.weight_epsilon = self.create_strain_weight_vector()

    def create_strain_weight_vector(self) -> np.ndarray:
        weights = []
        for i in range(int((len(self.config.PARAM_NAMES) - 1) / 2)):
            weights.extend([self.config.EPSILON_B, self.config.EPSILON_W])
        return np.array(weights)

    def apply_constraints(self, params_batch: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        params_np = params_batch.numpy().copy()

        numerator = (params_np[:, :-1] * self.weight_epsilon).sum(axis=1)
        denominator = params_np[:, :-1].sum(axis=1)
        strained_tol = numerator / denominator

        valid_indices = np.where(np.abs(strained_tol) < self.config.STRAIN_TOLERANCE)[0]

        if len(valid_indices) == 0:
            print("Warning: All suggested points violate strain constraints")
            return params_batch, strained_tol

        return params_batch[valid_indices], strained_tol[valid_indices]


