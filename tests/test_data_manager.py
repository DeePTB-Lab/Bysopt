import pytest
import pandas as pd
from bysopt.data_manager import DataManager


def test_load_experimental_data_valid(data_manager, sample_dataframe):
    df = data_manager.load_experimental_data()
    # print(df)

    assert df is not None
    assert len(df) == 3
    assert 'WPE' in df.columns
    assert 'Wavelength' in df.columns


def test_load_experimental_data_invalid_format(config, temp_data_file, temp_prediction_file):
    with open(temp_data_file, 'w') as f:
        f.write("invalid,column,names\n")
        f.write("1,2,3\n")

    dm = DataManager(config, temp_data_file, temp_prediction_file)
    df = dm.load_experimental_data()
    assert df is None


def test_load_experimental_data_empty(config, temp_data_file, temp_prediction_file):
    with open(temp_data_file, 'w') as f:
        pass

    dm = DataManager(config, temp_data_file, temp_prediction_file)
    df = dm.load_experimental_data()
    assert df is None


def test_load_experimental_data_nonexistent(config, temp_prediction_file):
    dm = DataManager(config, "nonexistent.csv", temp_prediction_file)
    df = dm.load_experimental_data()
    assert df is None


def test_save_predicted_parameters(data_manager, sample_parameters):
    suggestions = [
        {
            'parameters': sample_parameters,
            'net_strain': 0.05,
            'strain_violation': False
        }
    ]

    df = data_manager.save_predicted_parameters(suggestions)

    assert len(df) == 1
    assert df.iloc[0]['net_strain'] == 0.05
    assert df.iloc[0]['b1'] == sample_parameters['b1']


