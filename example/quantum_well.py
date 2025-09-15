import json
from bysopt.config import parse_args
from bysopt.optimization import QuantumWellOptimizationSystem

import warnings

warnings.filterwarnings('ignore')


def main():
    args = parse_args()
    optimization_system = QuantumWellOptimizationSystem(args)

    # Process any existing new data in CSV file
    has_new_data = optimization_system.process_existing_data()

    if has_new_data:
        print("Processed new experimental data, continuing optimization...")

    # Run optimization loop
    optimization_system.run_optimization_loop(max_iterations=args.max_iterations)

    best_params, best_wpe, best_wavelength, best_score = optimization_system.get_current_best()
    if best_params is not None:
        print(f"\nCurrent Best WPE: {best_wpe:.6f}, Wavelength: {best_wavelength:.4f} um")


if __name__ == "__main__":
    main()