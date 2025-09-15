import argparse

class QuantumWellConfig:
    def __init__(self):
        self.PARAM_BOUNDS = {
            'b1': (2.85, 4.75), 'w1': (1.2, 2),
            'b2': (0.95, 1.65), 'w2': (3.65, 6.15),
            'b3': (1.2, 2), 'w3': (2.55, 4.25),
            'b4': (2.05, 3.35), 'w4': (2.05, 3.35),
            'U': (110, 170)
        }
        self.PARAM_NAMES = list(self.PARAM_BOUNDS.keys())
        self.N_PARAMS = len(self.PARAM_NAMES)
        self.TARGET_WAVELENGTH = 4.0
        self.WPE_WEIGHT = 0.7
        self.WAVELENGTH_WEIGHT = 0.3
        self.EPSILON_B = -1.1008
        self.EPSILON_W = 0.8843
        self.STRAIN_TOLERANCE = 0.1
        self.THICKNESS_PRECISION = 0.5
        self.VOLTAGE_PRECISION = 3
        self.BATCH_SIZE = 3


def parse_args():
    parser = argparse.ArgumentParser(description='Quantum Well Optimization System')
    parser.add_argument('--max_iterations', type=int, default=20, help='Maximum number of optimization iterations')
    parser.add_argument('--data_file', type=str, default='../result/quantum_well_opt.csv', help='Path to experimental data file')
    parser.add_argument('--checkpoint', type=str, default='../result/bo_checkpoint.pt', help='Path to checkpoint')
    parser.add_argument('--prediction_file', type=str, default='../result/quantum_well_new.csv',
                        help='Path to prediction output file')
    parser.add_argument('--report_file', type=str, default='../result/optimization_report.json',
                        help='Path to optimization report file')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size for experiments')
    parser.add_argument('--target_wavelength', type=float, default=4.0, help='Target wavelength in micrometers')
    parser.add_argument('--wpe_weight', type=float, default=0.7, help='Weight for WPE in objective function')
    parser.add_argument('--wavelength_weight', type=float, default=0.3,
                        help='Weight for wavelength in objective function')

    return parser.parse_args()

