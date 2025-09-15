import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List, Optional
from bysopt.config import QuantumWellConfig


class DataManager:
    def __init__(self, config: QuantumWellConfig, data_file: str, prediction_file: str):
        self.config = config
        self.data_file = data_file
        self.prediction_file = prediction_file

    def load_experimental_data(self) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(self.data_file)
            required_columns = self.config.PARAM_NAMES + ['WPE', 'Wavelength']
            if all(col in df.columns for col in required_columns):
                return df
            else:
                print("Data file format incorrect")
                return None
        except FileNotFoundError:
            print("Data file not found, will create new one")
            return None
        except pd.errors.EmptyDataError:
            print("Data file is empty")
            return None


    def save_predicted_parameters(self, suggestions: List[Dict]) -> pd.DataFrame:
        data_to_save = []
        for suggestion in suggestions:
            row_data = {
                'net_strain': suggestion['net_strain'],
                'strain_violation': suggestion['strain_violation']
            }
            for param_name in self.config.PARAM_NAMES:
                row_data[param_name] = suggestion['parameters'][param_name]
            data_to_save.append(row_data)

        df_predicted = pd.DataFrame(data_to_save)
        columns_order = self.config.PARAM_NAMES + ['net_strain']
        df_predicted = df_predicted[columns_order]
        df_predicted.to_csv(self.prediction_file, index=False)
        return df_predicted

    def record_experiment_result(self, parameters: Dict, wpe: float, wavelength: float):
        row_data = {
            'WPE': wpe,
            'Wavelength': wavelength
        }

        for param_name in self.config.PARAM_NAMES:
            row_data[param_name] = parameters[param_name]

        try:
            df_existing = pd.read_csv(self.data_file)
            df_combined = pd.concat([df_existing, pd.DataFrame([row_data])], ignore_index=True)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            df_combined = pd.DataFrame([row_data])

        df_combined.to_csv(self.data_file, index=False)