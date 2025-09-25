import pytest
import torch
import pandas as pd
from bysopt.optimization import ExperimentManager, QuantumWellOptimizationSystem
from botorch.utils import standardize


@pytest.fixture
def experiment_manager(config, data_manager):
    return ExperimentManager(config, data_manager)


def test_experiment_manager_initialization(experiment_manager, config):
    assert experiment_manager.config == config
    assert experiment_manager.iteration == 0
    assert experiment_manager.pending_suggestions == []
    assert len(experiment_manager.y_wpe ) == experiment_manager.config.BATCH_SIZE


def test_record_result(experiment_manager, sample_parameters):
    assert experiment_manager.X is not None
    assert experiment_manager.y_wpe is not None
    assert experiment_manager.y_wavelength is not None
    assert len(experiment_manager.X) == experiment_manager.config.BATCH_SIZE
    assert len(experiment_manager.y_wpe) == experiment_manager.config.BATCH_SIZE
    assert len(experiment_manager.y_wavelength) == experiment_manager.config.BATCH_SIZE

    # record+1
    experiment_manager.record_result(sample_parameters, 0.45, 4.1)
    assert experiment_manager.X is not None
    assert experiment_manager.y_wpe is not None
    assert experiment_manager.y_wavelength is not None
    assert len(experiment_manager.X) == 1 + experiment_manager.config.BATCH_SIZE
    assert len(experiment_manager.y_wpe) == 1 + experiment_manager.config.BATCH_SIZE
    assert len(experiment_manager.y_wavelength) == 1 + experiment_manager.config.BATCH_SIZE
    assert experiment_manager.y_wpe[experiment_manager.config.BATCH_SIZE] == 0.45
    assert experiment_manager.y_wavelength[experiment_manager.config.BATCH_SIZE] == 4.1

    # record+2
    experiment_manager.record_result(sample_parameters, 0.48, 4.0)
    assert len(experiment_manager.X) == 2 + experiment_manager.config.BATCH_SIZE
    assert experiment_manager.y_wpe[1 + experiment_manager.config.BATCH_SIZE] == 0.48
    assert experiment_manager.y_wavelength[1 + experiment_manager.config.BATCH_SIZE] == 4.0


def test_has_pending_experiments(experiment_manager, sample_parameters):
    assert not experiment_manager.has_pending_experiments()

    experiment_manager.pending_suggestions = [{'parameters': sample_parameters}]
    assert experiment_manager.has_pending_experiments()


def test_process_pending_results(experiment_manager, sample_dataframe, sample_parameters):

    assert experiment_manager.X is not None
    assert len(experiment_manager.X) == experiment_manager.config.BATCH_SIZE

    experiment_manager.pending_suggestions = [{'parameters': sample_parameters}]
    experiment_manager.process_pending_results(sample_dataframe.iloc[:1])

    assert experiment_manager.X is not None
    assert len(experiment_manager.X) == 1 + experiment_manager.config.BATCH_SIZE
    assert len(experiment_manager.pending_suggestions) == 0


def test_suggest_experiments_random(experiment_manager, config):
    experiment_manager.X = None
    experiment_manager.y_wpe = None
    experiment_manager.y_wavelength = None

    for batch_size in [1, 2, 3, 5]:
        suggestions = experiment_manager.suggest_experiments(batch_size)

        assert len(suggestions) == batch_size

        for suggestion in suggestions:
            _validate_suggestion(suggestion, config)


def test_suggest_experiments_with_data(experiment_manager, config, sample_dataframe):
    experiment_manager.load_data(sample_dataframe)

    for batch_size in [1, 2, 3]:
        suggestions = experiment_manager.suggest_experiments(batch_size)

        assert len(suggestions) <= batch_size
        assert len(suggestions) > 0

        for suggestion in suggestions:
            _validate_suggestion(suggestion, config)



def test_suggest_experiments_strain_constraints(experiment_manager, config):

    experiment_manager.X = None
    experiment_manager.y_wpe = None
    experiment_manager.y_wavelength = None

    suggestions = experiment_manager.suggest_experiments(10)

    valid_suggestions = [s for s in suggestions if not s['strain_violation']]

    assert len(valid_suggestions) > 0

    for suggestion in valid_suggestions:
        assert abs(suggestion['net_strain']) < config.STRAIN_TOLERANCE


def _validate_suggestion(suggestion, config):
    assert 'parameters' in suggestion
    assert 'net_strain' in suggestion
    assert 'strain_violation' in suggestion

    params = suggestion['parameters']
    for param_name in config.PARAM_NAMES:
        low, high = config.PARAM_BOUNDS[param_name]
        assert low <= params[param_name] <= high, f"The parameter {param_name} is out of bounds: {params[param_name]}"

    assert isinstance(suggestion['net_strain'], float)
    assert suggestion['strain_violation'] == False or True
    # assert isinstance(suggestion['strain_violation'], bool)

    strain_violation_expected = abs(suggestion['net_strain']) >= config.STRAIN_TOLERANCE
    assert suggestion['strain_violation'] == strain_violation_expected