"""
MIT License
Copyright © 2024 David A. Reiss
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and
this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
"""
import argparse
import os
import sys

import json5
import numpy as np

from pathlib import Path

main_directory = Path(__file__).resolve().parent.parent
sys.path.append(str(main_directory))

from main import main
from config_template import Config
from logging_config import get_logger
from policy_evaluation_and_plots import PolicyEvaluation, plot_as_heatmap, convert_dict_to_data_frame, \
    plot_xy_vs_no_layers, plot_Fourier_coeffs
from reweighted_dynamics import ReweightedDynamics
from Fourier_series_analysis_and_fits import ParameterizedDynamicsFits, FourierSeriesAnalysis
from utilities import get_file_names_with_version, load_or_compute_obj, convert_to_and_save_latex_string, prepare_results_dir
from value_functions import ValueFunction


logger = get_logger("main.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Fourier series analysis for different time horizons T "
                                                 "with optional specification of config file.")
    parser.add_argument("--config_file_name", type=str, default="config_main_scaling_time_horizon.json5",
                        help="Path to the config file (default: config_main_scaling_time_horizon.json5)")
    args = parser.parse_args()

    with open(args.config_file_name, "r") as f:
        config_template = json5.load(f)

    for T in config_template["T"]:
        file_name = args.config_file_name.replace(".json5", f"_{T}.json5")
        config = config_template.copy()
        config["T"] = T

        with open(file_name, "w") as file:
            json5.dump(config, file, indent=4)

        print(f"Starting main for T = {T}...")
        main(file_name, create_plots=False)

        os.remove(file_name)



