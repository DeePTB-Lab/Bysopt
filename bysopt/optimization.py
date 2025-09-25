import torch
import pandas as pd
from typing import Dict, List, Optional, Tuple
from bysopt.config import QuantumWellConfig
from bysopt.utils import ResultAnalyzer
from bysopt.data_manager import DataManager
from bysopt.models import BayesianOptimizer
from bysopt.constraints import PhysicalConstraintProcessor
import numpy as np
import json
import os


class ExperimentManager:
    def __init__(self, config: QuantumWellConfig, data_manager: DataManager):
        self.config = config
        self.data_manager = data_manager
        self.optimizer = BayesianOptimizer(config)
        self.constraint_processor = PhysicalConstraintProcessor(config)

        self.X = None
        self.y_wpe = None
        self.y_wavelength = None
        self.iteration = 0
        self.pending_suggestions = []  # Store pending experiment suggestions

        self._initialize_data()

    def _initialize_data(self):
        df = self.data_manager.load_experimental_data()
        if df is not None:
            self.load_data(df)

    def load_data(self, data: pd.DataFrame):
        self.X = torch.tensor(data[self.config.PARAM_NAMES].values, dtype=torch.float32)
        self.y_wpe = torch.tensor(data['WPE'].values, dtype=torch.float32)
        self.y_wavelength = torch.tensor(data['Wavelength'].values, dtype=torch.float32)

    def suggest_experiments(self, batch_size: int = None) -> List[Dict]:
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE

        if self.X is None or len(self.X) < 2:
            candidates = self.optimizer.get_random_parameters(batch_size * 5)
        else:
            model_wpe = self.optimizer.create_surrogate_model(self.X, self.y_wpe)
            model_wavelength = self.optimizer.create_surrogate_model(self.X, self.y_wavelength)
            candidates = self.optimizer.optimize_acquisition_function(
                model_wpe, model_wavelength, self.X, self.y_wpe, self.y_wavelength, batch_size
            )

        constrained_candidates, strained_tol = self.constraint_processor.apply_constraints(candidates)

        if len(constrained_candidates) < batch_size:
            additional = self.optimizer.get_random_parameters(batch_size * 3)
            additional_constrained, additional_strained = self.constraint_processor.apply_constraints(additional)

            if len(additional_constrained) > 0:
                constrained_candidates = torch.cat([constrained_candidates, additional_constrained])
                strained_tol = np.concatenate([strained_tol, additional_strained])

        suggestions = []
        for i, params in enumerate(constrained_candidates[:batch_size]):
            param_dict = {name: params[j].item() for j, name in enumerate(self.config.PARAM_NAMES)}
            suggestions.append({
                'parameters': param_dict,
                'net_strain': strained_tol[i] * 0.01,
                'strain_violation': abs(strained_tol[i]) >= self.config.STRAIN_TOLERANCE
            })

        # Save current suggestions for later result recording
        self.pending_suggestions = suggestions.copy()

        return suggestions

    def record_result(self, parameters: Dict, wpe: float, wavelength: float):
        X_new = torch.tensor([parameters[param] for param in self.config.PARAM_NAMES], dtype=torch.float32)

        if self.X is None:
            self.X = X_new.unsqueeze(0)
            self.y_wpe = torch.tensor([wpe])
            self.y_wavelength = torch.tensor([wavelength])
        else:
            self.X = torch.cat([self.X, X_new.unsqueeze(0)])
            self.y_wpe = torch.cat([self.y_wpe, torch.tensor([wpe])])
            self.y_wavelength = torch.cat([self.y_wavelength, torch.tensor([wavelength])])



    def get_state_dict(self):
        """Get complete state of experiment manager"""
        state = {
            'iteration': self.iteration,
            'pending_suggestions': self.pending_suggestions,
        }

        if self.X is not None:
            state['X'] = self.X
            state['y_wpe'] = self.y_wpe
            state['y_wavelength'] = self.y_wavelength

        return state

    def load_state_dict(self, state_dict):
        """Restore experiment manager from state dictionary"""
        self.X = state_dict.get('X')
        self.y_wpe = state_dict.get('y_wpe')
        self.y_wavelength = state_dict.get('y_wavelength')
        self.iteration = state_dict.get('iteration', 0)
        self.pending_suggestions = state_dict.get('pending_suggestions', [])

    def has_pending_experiments(self):
        """Check if there are pending experiments"""
        return len(self.pending_suggestions) > 0

    def process_pending_results(self, results_data: pd.DataFrame):
        """Process results from pending experiments"""
        for _, row in results_data.iterrows():
            parameters = {param: row[param] for param in self.config.PARAM_NAMES}
            self.record_result(
                parameters,
                row['WPE'],
                row['Wavelength']
            )
        # Clear pending suggestions
        self.pending_suggestions = []


class QuantumWellOptimizationSystem:
    def __init__(self, args):
        self.config = QuantumWellConfig()
        self.update_config_from_args(args)

        self.data_manager = DataManager(self.config, args.data_file, args.prediction_file)
        self.experiment_manager = ExperimentManager(self.config, self.data_manager)
        self.result_analyzer = ResultAnalyzer(self.config)
        self.report_file = args.report_file

        self.checkpoint = args.checkpoint
        self.args = args

        self.load_checkpoint()

    def update_config_from_args(self, args):
        if args.batch_size:
            self.config.BATCH_SIZE = args.batch_size
        if args.target_wavelength:
            self.config.TARGET_WAVELENGTH = args.target_wavelength
        if args.wpe_weight:
            self.config.WPE_WEIGHT = args.wpe_weight
        if args.wavelength_weight:
            self.config.WAVELENGTH_WEIGHT = args.wavelength_weight

    def save_checkpoint(self):
        """Save complete optimization system state"""
        checkpoint_data = {
            'experiment_manager': self.experiment_manager.get_state_dict(),
            'config': self.config.__dict__,
            'args': vars(self.args)
        }
        torch.save(checkpoint_data, self.checkpoint)
        print(f"Checkpoint saved to {self.checkpoint}")

    def load_checkpoint(self):
        """Restore optimization system state from checkpoint"""
        if os.path.exists(self.checkpoint):
            try:
                checkpoint_data = torch.load(self.checkpoint, weights_only=False)
                self.experiment_manager.load_state_dict(checkpoint_data['experiment_manager'])
                print(f"Checkpoint loaded from {self.checkpoint}, iteration: {self.experiment_manager.iteration}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")

    def run_optimization_loop(self, max_iterations: int = 20):
        if self.experiment_manager.has_pending_experiments():
            print("Found pending experiments from previous run.")
            print("Please add experimental results to CSV file and run again.")
            return

        print("Starting Quantum Well Optimization System...")
        start_iteration = self.experiment_manager.iteration
        for iteration in range(start_iteration, max_iterations):
            print(f"\n=== Optimization Iteration {iteration + 1}/{max_iterations} ===")

            # Generate new experiment suggestions
            suggestions = self.experiment_manager.suggest_experiments()
            self.data_manager.save_predicted_parameters(suggestions)

            # Save checkpoint (after generating suggestions)
            self.experiment_manager.iteration += 1
            self.save_checkpoint()

            print(f"Generated {len(suggestions)} new experiment suggestions")
            print("Suggested parameters saved to:", self.args.prediction_file)
            print("Please run experiments and add results to:", self.args.data_file)
            print("Then run the program again to continue optimization")

            # Auto-stop after generating suggestions for async execution
            break

        # Generate final report if all iterations completed
        if self.experiment_manager.iteration >= max_iterations:
            self.generate_final_report()

    def process_existing_data(self):
        """Process any new data in CSV file"""
        df = self.data_manager.load_experimental_data()
        if df is not None and len(df) > 0:
            # Get current data count
            current_count = 0
            if self.experiment_manager.X is not None:
                current_count = len(self.experiment_manager.X)

            # Process new data if available
            if len(df) > current_count:
                new_data = df.iloc[current_count:]
                print(f"Found {len(new_data)} new experimental results")

                self.experiment_manager.process_pending_results(new_data)
                print(f"Processed {len(new_data)} new results, total iterations: {self.experiment_manager.iteration}")
                return True
        return False

    def get_current_best(self):
        if self.experiment_manager.X is None:
            return None, None, None, None

        return self.result_analyzer.get_best_result(
            self.experiment_manager.X,
            self.experiment_manager.y_wpe,
            self.experiment_manager.y_wavelength
        )

    def generate_final_report(self):
        best_params, best_wpe, best_wavelength, best_score = self.get_current_best()

        if best_params is not None:
            report = self.result_analyzer.generate_report(
                best_params, best_wpe, best_wavelength, best_score
            )

            print("\n" + "=" * 60)
            print("Quantum Well Optimization Final Report")
            print("=" * 60)
            print(f"Best Combined Score: {best_score:.4f}")
            print(f"Best WPE: {best_wpe:.6f}")
            print(f"Best Wavelength: {best_wavelength:.4f} um (Target: {self.config.TARGET_WAVELENGTH} um)")
            print("\nBest Parameters:")
            for name, value in report['best_parameters'].items():
                print(f"  {name}: {value:.4f}")

            with open(self.report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nDetailed report saved to: {self.report_file}")
