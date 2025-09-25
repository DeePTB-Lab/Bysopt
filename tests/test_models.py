import pytest
import torch
import gpytorch
import numpy as np
from bysopt.models import BayesianOptimizer
from botorch.models import SingleTaskGP



class TestBayesianOptimizer:

    def test_initialization(self, config):
        optimizer = BayesianOptimizer(config)
        assert optimizer.config == config
        assert optimizer.param_bounds.shape == (2, config.N_PARAMS)
        assert optimizer.param_bounds.dtype == torch.float32

    def test_get_random_parameters_shape(self, config):
        optimizer = BayesianOptimizer(config)

        for n_samples in [1, 5, 10, 20]:
            params = optimizer.get_random_parameters(n_samples)
            assert params.shape == (n_samples, config.N_PARAMS)
            assert params.dtype == torch.float32


    def test_get_random_parameters_bounds(self, config):
        optimizer = BayesianOptimizer(config)
        n_samples = 1000

        params = optimizer.get_random_parameters(n_samples)

        for i, param_name in enumerate(config.PARAM_NAMES):
            low, high = config.PARAM_BOUNDS[param_name]

            assert torch.all(params[:, i] >= low - 1e-6)
            assert torch.all(params[:, i] <= high + 1e-6)

            param_range = params[:, i]
            assert (torch.max(param_range) - torch.min(param_range)) > (high - low) * 0.8


    def test_create_surrogate_model_basic(self, config, training_data):
        optimizer = BayesianOptimizer(config)
        X, y_wpe, y_wavelength = training_data

        X = X.double()
        y_wpe = y_wpe.double()

        model = optimizer.create_surrogate_model(X, y_wpe)

        assert isinstance(model, SingleTaskGP)
        assert hasattr(model, 'train_inputs')
        assert hasattr(model, 'train_targets')
        assert hasattr(model, 'likelihood')
        assert hasattr(model, 'train')
        assert hasattr(model, 'eval')

        assert len(model.train_inputs) == 1
        assert model.train_inputs[0].shape == X.shape
        assert model.train_targets.shape == y_wpe.shape

    def test_create_surrogate_model_prediction(self, config, training_data):
        optimizer = BayesianOptimizer(config)
        X, y_wpe, y_wavelength = training_data

        X = X.double()
        y_wpe = y_wpe.float()

        model = optimizer.create_surrogate_model(X, y_wpe)
        model.eval()

        test_X = torch.tensor([
            [3.2, 1.55, 1.25, 4.2, 1.55, 3.2, 2.55, 2.55, 145],
            [3.8, 1.65, 1.35, 4.8, 1.65, 3.8, 2.65, 2.65, 155]
        ], dtype=torch.double)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = model(test_X)
            predictions = posterior.mean

            assert predictions is not None
            assert predictions.shape[0] == 2
            assert not torch.any(torch.isnan(predictions))
            assert not torch.any(torch.isinf(predictions))


    def test_optimize_acquisition_function_basic(self, config, training_data):
        optimizer = BayesianOptimizer(config)
        X, y_wpe, y_wavelength = training_data

        X = X.double()
        y_wpe = y_wpe.double()
        y_wavelength = y_wavelength.double()

        model_wpe = optimizer.create_surrogate_model(X, y_wpe)
        model_wavelength = optimizer.create_surrogate_model(X, y_wavelength)

        n_candidates = 2
        candidates = optimizer.optimize_acquisition_function(
            model_wpe, model_wavelength, X, y_wpe, y_wavelength, n_candidates
        )

        assert candidates is not None
        assert candidates.shape == (n_candidates, config.N_PARAMS)
        assert candidates.dtype == torch.float32

        for i in range(config.N_PARAMS):
            low, high = config.PARAM_BOUNDS[config.PARAM_NAMES[i]]
            assert torch.all(candidates[:, i] >= low - 1e-6)
            assert torch.all(candidates[:, i] <= high + 1e-6)



    def test_full_optimization_cycle(self, config, training_data):
        optimizer = BayesianOptimizer(config)
        X, y_wpe, y_wavelength = training_data

        X = X.double()
        y_wpe = y_wpe.double()
        y_wavelength = y_wavelength.double()

        #step1
        model_wpe = optimizer.create_surrogate_model(X, y_wpe)
        model_wavelength = optimizer.create_surrogate_model(X, y_wavelength)

        # step2
        n_candidates = 3
        candidates = optimizer.optimize_acquisition_function(
            model_wpe, model_wavelength, X, y_wpe, y_wavelength, n_candidates
        )

        # step3
        assert candidates.shape == (n_candidates, config.N_PARAMS)

        # step4
        for i in range(config.N_PARAMS):
            low, high = config.PARAM_BOUNDS[config.PARAM_NAMES[i]]
            assert torch.all(candidates[:, i] >= low - 1e-6)
            assert torch.all(candidates[:, i] <= high + 1e-6)


    def test_model_training_mode_switching(self, config, training_data):
        optimizer = BayesianOptimizer(config)
        X, y_wpe, y_wavelength = training_data

        X = X.double()
        y_wpe = y_wpe.double()

        model = optimizer.create_surrogate_model(X, y_wpe)

        model.train()
        assert model.training

        model.eval()
        assert not model.training

        model.train()
        assert model.training

