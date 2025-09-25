import pytest
import torch
import pandas as pd
import tempfile
import sys
import os
sys.path.append((os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))))

from bysopt.config import QuantumWellConfig
from bysopt.data_manager import DataManager


@pytest.fixture
def config():
    return QuantumWellConfig()


@pytest.fixture
def sample_dataframe(config):
    data = {
        'b1': [3.0, 3.5, 4.0],
        'w1': [1.5, 1.6, 1.7],
        'b2': [1.2, 1.3, 1.4],
        'w2': [4.0, 4.5, 5.0],
        'b3': [1.5, 1.6, 1.7],
        'w3': [3.0, 3.5, 4.0],
        'b4': [2.5, 2.6, 2.7],
        'w4': [2.5, 2.6, 2.7],
        'U': [140, 150, 160],
        'WPE': [0.45, 0.48, 0.52],
        'Wavelength': [4.1, 4.0, 3.9]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_parameters():
    return {
        'b1': 3.0, 'w1': 1.5,
        'b2': 1.2, 'w2': 4.0,
        'b3': 1.5, 'w3': 3.0,
        'b4': 2.5, 'w4': 2.5,
        'U': 140
    }


@pytest.fixture
def temp_data_file(config,sample_dataframe):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_dataframe.to_csv(f.name, index=False)
        temp_file = f.name

    yield temp_file
    os.unlink(temp_file)


@pytest.fixture
def temp_prediction_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_file = f.name

    yield temp_file
    # print(temp_file)
    if os.path.exists(temp_file):
        os.unlink(temp_file)


@pytest.fixture
def data_manager(config, temp_data_file, temp_prediction_file):
    return DataManager(config, temp_data_file, temp_prediction_file)



@pytest.fixture
def training_data(config):
    X = torch.tensor([
        [3.0, 1.5, 1.2, 4.0, 1.5, 3.0, 2.5, 2.5, 140],
        [3.5, 1.6, 1.3, 4.5, 1.6, 3.5, 2.6, 2.6, 150],
        [4.0, 1.7, 1.4, 5.0, 1.7, 4.0, 2.7, 2.7, 160],
        [3.2, 1.55, 1.25, 4.2, 1.55, 3.2, 2.55, 2.55, 145],
        [3.8, 1.65, 1.35, 4.8, 1.65, 3.8, 2.65, 2.65, 155]
    ], dtype=torch.float64)

    y_wpe = torch.tensor([0.45, 0.48, 0.52, 0.47, 0.50], dtype=torch.float64)
    y_wavelength = torch.tensor([4.1, 4.0, 3.9, 4.05, 3.95], dtype=torch.float64)

    return X, y_wpe, y_wavelength