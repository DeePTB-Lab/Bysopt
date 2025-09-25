import pytest
import torch
import pandas as pd
import numpy as np
import os
import json
import tempfile
from bysopt.optimization import QuantumWellOptimizationSystem
from bysopt.config import QuantumWellConfig, parse_args


class TestQuantumWellOptimizationSystemIntegration:

    @pytest.fixture
    def system_args(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as data_file:
            data_file.write("b1,w1,b2,w2,b3,w3,b4,w4,U,WPE,Wavelength\n")
            data_file.write("3.0,1.5,1.2,4.0,1.5,3.0,2.5,2.5,140,0.45,4.1\n")
            data_file_name = data_file.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as pred_file:
            pred_file_name = pred_file.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as report_file:
            report_file_name = report_file.name

        # with tempfile.NamedTemporaryFile(mode='w', suffix='.pt', delete=False) as checkpoint_file:
        #     checkpoint_file_name = checkpoint_file.name
        checkpoint_file_name='bo_checkpoint.pt'

        args = type('Args', (), {
            'data_file': data_file_name,
            'prediction_file': pred_file_name,
            'checkpoint': checkpoint_file_name,
            'report_file': report_file_name,
            'max_iterations': 20,
            'batch_size': 3,
            'target_wavelength': 4.0,
            'wpe_weight': 0.7,
            'wavelength_weight': 0.3
        })()

        yield args

        for file_path in [data_file_name, pred_file_name, report_file_name, checkpoint_file_name]:
            if os.path.exists(file_path):
                os.unlink(file_path)

    def test_system_initialization(self, system_args):
        system = QuantumWellOptimizationSystem(system_args)

        assert system.config is not None
        assert isinstance(system.config, QuantumWellConfig)
        assert system.data_manager is not None
        assert system.experiment_manager is not None
        assert system.result_analyzer is not None

        assert system.config.BATCH_SIZE == 3
        assert system.config.TARGET_WAVELENGTH == 4.0
        assert system.config.WPE_WEIGHT == 0.7
        assert system.config.WAVELENGTH_WEIGHT == 0.3

    def test_update_config_from_args(self, system_args):
        system = QuantumWellOptimizationSystem(system_args)
        original_config = {
            'BATCH_SIZE': system.config.BATCH_SIZE,
            'TARGET_WAVELENGTH': system.config.TARGET_WAVELENGTH,
            'WPE_WEIGHT': system.config.WPE_WEIGHT,
            'WAVELENGTH_WEIGHT': system.config.WAVELENGTH_WEIGHT
        }

        custom_args = type('Args', (), {
            'batch_size': 5,
            'target_wavelength': 4.2,
            'wpe_weight': 0.6,
            'wavelength_weight': 0.4,
            'data_file': system_args.data_file,
            'prediction_file': system_args.prediction_file,
            'checkpoint': system_args.checkpoint,
            'report_file': system_args.report_file,
            'max_iterations': 20
        })()

        system.update_config_from_args(custom_args)

        assert system.config.BATCH_SIZE == 5
        assert system.config.TARGET_WAVELENGTH == 4.2
        assert system.config.WPE_WEIGHT == 0.6
        assert system.config.WAVELENGTH_WEIGHT == 0.4

        system.config.BATCH_SIZE = original_config['BATCH_SIZE']
        system.config.TARGET_WAVELENGTH = original_config['TARGET_WAVELENGTH']
        system.config.WPE_WEIGHT = original_config['WPE_WEIGHT']
        system.config.WAVELENGTH_WEIGHT = original_config['WAVELENGTH_WEIGHT']

    def test_process_existing_data_with_data(self, system_args):
        system = QuantumWellOptimizationSystem(system_args)

        assert system.experiment_manager.X is not None
        assert system.experiment_manager.y_wpe is not None
        assert system.experiment_manager.y_wavelength is not None
        assert len(system.experiment_manager.X) == 1

        expected_params = torch.tensor([3.0, 1.5, 1.2, 4.0, 1.5, 3.0, 2.5, 2.5, 140], dtype=torch.float32)
        assert torch.allclose(system.experiment_manager.X[0], expected_params)
        assert system.experiment_manager.y_wpe[0] == pytest.approx(0.45)
        assert system.experiment_manager.y_wavelength[0] == pytest.approx(4.1)

    def test_process_existing_data_no_new_data(self, system_args):
        system = QuantumWellOptimizationSystem(system_args)

        has_new_data = system.process_existing_data()
        assert not has_new_data

    def test_get_current_best_with_data(self, system_args):
        system = QuantumWellOptimizationSystem(system_args)
        best_params, best_wpe, best_wavelength, best_score = system.get_current_best()

        assert best_params is not None
        assert best_wpe is not None
        assert best_wavelength is not None
        assert best_score is not None

        for i, param_name in enumerate(system.config.PARAM_NAMES):
            low, high = system.config.PARAM_BOUNDS[param_name]
            assert low <= best_params[i] <= high


    def test_save_and_load_checkpoint(self, system_args):
        system = QuantumWellOptimizationSystem(system_args)

        system.experiment_manager.iteration = 5
        system.experiment_manager.pending_suggestions = [{'test': 'data'}]

        system.save_checkpoint()

        assert os.path.exists(system_args.checkpoint)

        new_system = QuantumWellOptimizationSystem(system_args)
        new_system.load_checkpoint()

        assert new_system.experiment_manager.iteration == 5
        assert new_system.experiment_manager.pending_suggestions == [{'test': 'data'}]
        assert torch.allclose(new_system.experiment_manager.X, system.experiment_manager.X)
        assert torch.allclose(new_system.experiment_manager.y_wpe, system.experiment_manager.y_wpe)
        assert torch.allclose(new_system.experiment_manager.y_wavelength, system.experiment_manager.y_wavelength)

    def test_generate_final_report(self, system_args):
        system = QuantumWellOptimizationSystem(system_args)
        system.generate_final_report()
        assert os.path.exists(system_args.report_file)
        with open(system_args.report_file, 'r') as f:
            report = json.load(f)

        assert 'best_parameters' in report
        assert 'performance_metrics' in report
        assert 'strain_info' in report
        assert 'optimization_config' in report

        best_params = report['best_parameters']
        for param_name in system.config.PARAM_NAMES:
            assert param_name in best_params
            low, high = system.config.PARAM_BOUNDS[param_name]
            assert low <= best_params[param_name] <= high

    def test_full_optimization_workflow(self, system_args):
        system = QuantumWellOptimizationSystem(system_args)
        assert system

        best_params, best_wpe, best_wavelength, best_score = system.get_current_best()
        assert best_params is not None

        system.run_optimization_loop(max_iterations=1)

        assert os.path.exists(system_args.prediction_file)
        predictions_df = pd.read_csv(system_args.prediction_file)
        print(predictions_df)
        assert len(predictions_df) > 0

        expected_columns = system.config.PARAM_NAMES + ['net_strain']
        assert all(col in predictions_df.columns for col in expected_columns)

        for _, row in predictions_df.iterrows():
            for param_name in system.config.PARAM_NAMES:
                low, high = system.config.PARAM_BOUNDS[param_name]
                assert low <= row[param_name] <= high

        assert os.path.exists(system_args.checkpoint)
