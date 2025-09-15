import torch
import json
from typing import Dict, Tuple
from bysopt.config import QuantumWellConfig


class ResultAnalyzer:
    def __init__(self, config: QuantumWellConfig):
        self.config = config

    def calculate_scores(self, wpe_values: torch.Tensor, wavelength_values: torch.Tensor) -> torch.Tensor:
        if len(wpe_values) == 0:
            return torch.tensor([])

        wpe_score = wpe_values / wpe_values.max()
        wavelength_score = 1.0 / (1.0 + torch.abs(wavelength_values - self.config.TARGET_WAVELENGTH))
        return (self.config.WPE_WEIGHT * wpe_score +
                self.config.WAVELENGTH_WEIGHT * wavelength_score)

    def get_best_result(self, X: torch.Tensor, y_wpe: torch.Tensor, y_wavelength: torch.Tensor) -> Tuple:
        if X is None or len(X) == 0:
            return None, None, None, None

        scores = self.calculate_scores(y_wpe, y_wavelength)
        best_idx = scores.argmax()

        return (
            X[best_idx],
            y_wpe[best_idx].item(),
            y_wavelength[best_idx].item(),
            scores[best_idx].item()
        )

    def generate_report(self, best_params: torch.Tensor, best_wpe: float,
                        best_wavelength: float, best_score: float) -> Dict:
        return {
            'best_parameters': {name: best_params[i].item() for i, name in enumerate(self.config.PARAM_NAMES)},
            'performance_metrics': {
                'WPE': best_wpe,
                'Wavelength': best_wavelength,
                'Target_Wavelength': self.config.TARGET_WAVELENGTH,
                'Combined_Score': best_score
            },
            'strain_info': {
                'epsilon_barrier': self.config.EPSILON_B,
                'epsilon_well': self.config.EPSILON_W,
                'tolerance': self.config.STRAIN_TOLERANCE
            },
            'optimization_config': {
                'wpe_weight': self.config.WPE_WEIGHT,
                'wavelength_weight': self.config.WAVELENGTH_WEIGHT
            }
        }
