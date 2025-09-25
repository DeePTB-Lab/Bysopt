import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils import standardize
from typing import Tuple
from bysopt.config import QuantumWellConfig
from botorch.models.transforms.input import Normalize

class BayesianOptimizer:
    def __init__(self, config: QuantumWellConfig):
        self.config = config
        self.param_bounds = torch.tensor([list(bounds) for bounds in config.PARAM_BOUNDS.values()]).T

    def create_surrogate_model(self, X: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
        y_normalized = standardize(y.unsqueeze(-1)).type(torch.double)
        X = X.type(torch.double)
        model = SingleTaskGP(X, y_normalized,input_transform=Normalize(d=X.shape[-1]))
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def optimize_acquisition_function(self, model_wpe: SingleTaskGP,
                                      model_wavelength: SingleTaskGP,
                                      X_train: torch.Tensor,
                                      y_wpe: torch.Tensor,
                                      y_wavelength: torch.Tensor,
                                      n_candidates: int) -> torch.Tensor:
        def combined_acquisition(X):
            ei_wpe = qExpectedImprovement(model_wpe, best_f=y_wpe.max())
            ei_wavelength = qExpectedImprovement(
                model_wavelength,
                best_f=(-torch.abs(y_wavelength - self.config.TARGET_WAVELENGTH)).max()
            )
            return (self.config.WPE_WEIGHT * ei_wpe(X) +
                    self.config.WAVELENGTH_WEIGHT * ei_wavelength(X))

        candidates, _ = optimize_acqf(
            acq_function=combined_acquisition,
            bounds=self.param_bounds,
            q=n_candidates,
            num_restarts=10,
            raw_samples=100,
        )
        return candidates

    def get_random_parameters(self, n_samples: int) -> torch.Tensor:
        candidates = torch.rand(n_samples, self.config.N_PARAMS)
        for i in range(self.config.N_PARAMS):
            low, high = self.param_bounds[0, i], self.param_bounds[1, i]
            candidates[:, i] = low + (high - low) * candidates[:, i]
        return candidates