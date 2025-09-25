import numpy as np
import pytest
import torch
from bysopt.utils import ResultAnalyzer


def test_calculate_scores(config):
    analyzer = ResultAnalyzer(config)

    wpe_values = torch.tensor([0.4, 0.5, 0.6])
    wavelength_values = torch.tensor([4.1, 4.0, 3.9])

    scores = analyzer.calculate_scores(wpe_values, wavelength_values)

    assert len(scores) == 3
    assert all(0 <= score <= 1 for score in scores)
    assert scores[2] == scores.max()


def test_get_best_result(config,training_data):
    analyzer = ResultAnalyzer(config)
    X, y_wpe, y_wavelength = training_data

    best_params, best_wpe, best_wavelength, best_score = analyzer.get_best_result(X, y_wpe, y_wavelength)

    assert np.round(best_wpe,2) == 0.52
    assert np.round(best_wavelength,2) == 3.9
    assert best_score > 0


def test_get_best_result_empty(config):
    analyzer = ResultAnalyzer(config)

    result = analyzer.get_best_result(None, None, None)
    assert result == (None, None, None, None)

    result = analyzer.get_best_result(torch.tensor([]), torch.tensor([]), torch.tensor([]))
    assert result == (None, None, None, None)


def test_generate_report(config, sample_parameters):
    analyzer = ResultAnalyzer(config)

    best_params = torch.tensor([sample_parameters[name] for name in config.PARAM_NAMES])
    report = analyzer.generate_report(best_params, 0.52, 3.9, 0.85)

    assert 'best_parameters' in report
    assert 'performance_metrics' in report
    assert 'strain_info' in report
    assert report['performance_metrics']['WPE'] == 0.52
    assert report['performance_metrics']['Wavelength'] == 3.9