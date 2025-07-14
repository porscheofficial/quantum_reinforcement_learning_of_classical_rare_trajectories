"""
MIT License
Copyright © 2025 David A. Reiss
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
import copy
import json
import math
import warnings
import json5
import pytest
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Any
from pathlib import Path

main_directory = Path(__file__).resolve().parent.parent
sys.path.append(str(main_directory))

from config_template import Config
from reweighted_dynamics import ReweightedDynamics
from Fourier_series_analysis_and_fits import ParameterizedDynamicsFits
from utilities import save_obj, prepare_results_dir, get_file_names_with_version, InfoMessage


class BenchmarkUtilities:
    """
    Utility class for benchmarking and plotting results of scaling benchmarks for reweighted dynamics and
    parameterized dynamics fits.
    """

    @staticmethod
    def _compute_reweighted_dynamics(config: Config) -> ReweightedDynamics:
        return ReweightedDynamics(config.T, config.s, config.x_T, config.prob_step_up)


    @staticmethod
    def _fit_parameterized_dynamics_to_required_accuracy(file_name_list: list[str], reweighted_dynamics: ReweightedDynamics,
                                                        config: Config, required_min_residual_cost: float,
                                                        save_result: bool, compute_in_parallel: bool) -> float:
            assert len(config.no_qubits_list) == 1, "len(config.no_qubits_list) == 1 required"
            assert len(config.no_layers_list) == 1, "len(config.no_layers_list) == 1 required"

            fitting = True
            current_no_fits = 0

            while fitting:
                current_no_fits += 1

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", InfoMessage)

                    for file_name in file_name_list:
                        parameterized_dynamics_fits = ParameterizedDynamicsFits(
                            reweighted_dynamics.reweighted_dynamics_P_W,
                            config.no_qubits_list[0], config.no_layers_list[0], config.no_fits,
                            config.fitting_parameters, config.cost_func_type,
                            no_trajectories_cost_func=config.no_trajectories_cost_func,
                            max_optimization_steps=config.max_optimization_steps,
                            #no_random_Fourier_features=config.no_random_Fourier_features,
                            T=config.T, s=config.s, x_T=config.x_T, prob_step_up=config.prob_step_up,
                            optimal_average_return=np.log(reweighted_dynamics.partition_function_Z),
                            compute_in_parallel=False
                            # compute_in_parallel=False import because in method fit_policy of that class
                            # the no_fits-many fitting processes are parallelized
                        )

                        for info_message in w:
                            if issubclass(info_message.category, InfoMessage):
                                min_residual_cost = float(str(info_message.message).split("min_residual_cost = ")[-1])

                                if min_residual_cost < required_min_residual_cost:
                                    fitting = False

            if save_result:
                save_obj(parameterized_dynamics_fits, file_name)

            return min_residual_cost


    @staticmethod
    def _get_config_benchmark(config_file_name: str) -> dict[str, Any]:
        with open(config_file_name, "r") as f:
            config_benchmark = json5.load(f)
        return config_benchmark


    @staticmethod
    def _get_config(config_benchmark: dict[str, Any], T: int) -> Config:
        config_fits = copy.deepcopy(config_benchmark)
        config_fits["T"] = T
        del config_fits["no_fits_first_T_value"]  # to avoid ValidationError for class Config
        config_fits["no_fits"] = 1  # instead of a fixed value of fits as in the first run, function
                                    # fit_parameterized_dynamics_to_required_accuracy() below will fit as many times
                                    # as necessary to reach accuracy of first run
        config_fits["no_samples_variational_params"] = 1  # no relevance here, only set to avoid ValidationError for class Config
        config_fits["no_trajectories_policy_evaluation"] = 1  # --"--
        config_fits["policy_selection_criterion"] = "max_prob_rare_trajectory"  # --"--
        return Config(**config_fits)


    @staticmethod
    def _load_results(file_name: str) -> pd.DataFrame:
        with open(file_name) as f:
            data = json.load(f)

        df_dict = {"T": [], "mean_time": [], "std_dev_time": [], "rounds": [], "x_tick_label": []}

        for benchmark in data['benchmarks']:
            df_dict["T"].append(benchmark["params"]["T"])
            df_dict["mean_time"].append(benchmark['stats']['mean'])
            df_dict["std_dev_time"].append(benchmark['stats']['stddev'])
            df_dict["rounds"].append(benchmark['stats']['rounds'])

            df_dict["x_tick_label"].append(f"{benchmark['params']['T']} [{benchmark['stats']['rounds']}]")

        return pd.DataFrame(df_dict)


    @staticmethod
    def _plot_results(df: pd.DataFrame, plot_file_name: str, title: str, yaxis_type='log', num_ticks=7) -> None:
        assert yaxis_type in ['log', 'linear']

        fig = go.Figure(go.Scatter(x=df['T'], y=df['mean_time'],
                                   error_y=dict(type='data', array=df['std_dev_time'], visible=(yaxis_type == "linear")),
                                   mode='markers'))

        if yaxis_type == 'linear':
            factor = 10 ** math.floor(math.log10(df['mean_time'].max()))
            yaxis_tickvals = np.linspace(0., math.ceil(df['mean_time'].max() / factor) * factor,
                                         num=num_ticks, endpoint=True)

            yaxis_ticktext = [f"{val:.1e}" for val in yaxis_tickvals]

        if yaxis_type == 'log':
            yaxis_tickvals = [10 ** math.ceil(math.log10(df['mean_time'].max()))]

            for i in range(num_ticks - 1):
                if i % 2 == 0:
                    yaxis_tickvals.append(yaxis_tickvals[-1] / 2)
                else:
                    yaxis_tickvals.append(yaxis_tickvals[-1] / 5)

            yaxis_ticktext = [f"{val:.1e}" for val in yaxis_tickvals]

        fig.update_layout(xaxis_type='log',
                          xaxis=dict(tickvals=df['T'], ticktext=df['x_tick_label'],
                                     linecolor='black', tickcolor='black', gridcolor='black', griddash='dot',
                                     zerolinecolor='black'),
                          xaxis_title=r"$\text{random walk time horizon}~T~\text{(steps) [# benchmarks]}$",
                          yaxis_type=yaxis_type,
                          yaxis=dict(tickvals=yaxis_tickvals, ticktext=yaxis_ticktext,
                                     linecolor='black', tickcolor='black', gridcolor='black', griddash='dot',
                                     zerolinecolor='black'),
                          yaxis_title=r"$\text{computation time (seconds)}$",
                          title=title,
                          font=dict(family="serif", size=14),
                          plot_bgcolor='white', paper_bgcolor='white')

        fig.write_image(plot_file_name)


# fixtures
@pytest.fixture(scope="session")
def config_reweighted_dynamics_benchmark():
    return BenchmarkUtilities._get_config_benchmark("config_reweighted_dynamics_scaling_time_horizon.json5")


@pytest.fixture(scope="session")
def config_parameterized_dynamics_fits_benchmark():
    return BenchmarkUtilities._get_config_benchmark("config_parameterized_dynamics_fits_scaling_time_horizon.json5")


@pytest.fixture(scope="session")
def required_min_residual_cost(config_parameterized_dynamics_fits_benchmark):
    config = BenchmarkUtilities._get_config(config_parameterized_dynamics_fits_benchmark,
                                            config_parameterized_dynamics_fits_benchmark["T"][0])
    reweighted_dynamics = BenchmarkUtilities._compute_reweighted_dynamics(config)

    # set required min. residual cost to min. residual cost for first T value and no_fits_first_T_value-many fits
    config.no_fits = config_parameterized_dynamics_fits_benchmark["no_fits_first_T_value"]
    min_residual_cost = BenchmarkUtilities._fit_parameterized_dynamics_to_required_accuracy(
        ["mock_name"], reweighted_dynamics, config, np.inf, save_result=False, compute_in_parallel=True)
    return min_residual_cost


# tests
@pytest.mark.parametrize("T", BenchmarkUtilities._get_config_benchmark("config_reweighted_dynamics_scaling_time_horizon.json5")["T"])
def test_reweighted_dynamics_scaling_time_horizon(T, config_reweighted_dynamics_benchmark, benchmark):
    config = BenchmarkUtilities._get_config(config_reweighted_dynamics_benchmark, T)

    benchmark(lambda: BenchmarkUtilities._compute_reweighted_dynamics(config))


@pytest.mark.parametrize("T", BenchmarkUtilities._get_config_benchmark("config_parameterized_dynamics_fits_scaling_time_horizon.json5")["T"])
def test_parameterized_dynamics_fits_scaling_time_horizon(T, config_parameterized_dynamics_fits_benchmark,
                                                          required_min_residual_cost, benchmark,
                                                          save_result=True):
    """
    Benchmark computation time for fitting parameterized dynamics to reweighted dynamics P_W.

    Parameters:
        T: time horizon of random walk
        config_benchmark: configuration for benchmark, provided as a fixture
        required_min_residual_cost: minimum residual cost for fits, provided as a fixture
        benchmark: pytest-benchmark fixture to measure time taken for computation
        save_result: whether to save result of fits to a file;
            NOTE that if True this might slightly distort the benchmark time
    """

    # TODO: clarify whether different seeds for np.random should be used for each T value

    benchmark_utils = BenchmarkUtilities

    config = benchmark_utils._get_config(config_parameterized_dynamics_fits_benchmark, T)

    reweighted_dynamics = benchmark_utils._compute_reweighted_dynamics(config)

    path_computations, path_plots = prepare_results_dir(f"config_parameterized_dynamics_fits_scaling_time_horizon_{T}.json5",
                                                        dump_dict=config.__dict__)

    Path(path_plots).rmdir()

    if config.fitting_parameters in ("Fourier_coefficients", "variational_angles"):
        file_name_list = [f"{path_computations}/fits_qubits_{config.no_qubits_list[0]}_layers_{config.no_layers_list[0]}"
                          f"_{config.cost_func_type}_fitting_parameters_{config.fitting_parameters}.npz"]

    if config.fitting_parameters == "random_Fourier_features":
        file_name = (f"{path_computations}/fits_qubits_{config.no_qubits_list[0]}_layers_{config.no_layers_list[0]}"
                     f"_{config.cost_func_type}_random_Fourier_features_{config.no_random_Fourier_features}.npz")

        file_name_list, _ = get_file_names_with_version(file_name, config.no_choices_random_Fourier_features,
                                                        path_computations)

    benchmark(lambda: benchmark_utils._fit_parameterized_dynamics_to_required_accuracy(file_name_list, reweighted_dynamics,
                                                                                       config, required_min_residual_cost,
                                                                                       save_result,
                                                                                       compute_in_parallel=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scaling benchmarks with optional specification of benchmark.")
    parser.add_argument("--benchmarks", type=str, default="all",
                        help="Benchmarks to be executed (default: 'all'; options: 'all', "
                             "'test_reweighted_dynamics_scaling_time_horizon', "
                             "'test_parameterized_dynamics_fits_scaling_time_horizon')")
    benchmarks = parser.parse_args().benchmarks
    benchmark_utils = BenchmarkUtilities


    # run scaling benchmarks and plot results if this script is executed directly
    if benchmarks == "all" or benchmarks == "test_reweighted_dynamics_scaling_time_horizon":
        print("Running benchmark for reweighted dynamics scaling with time horizon...")

        pytest.main([__file__,
                     "-k", "test_reweighted_dynamics_scaling_time_horizon",
                     "--benchmark-enable", "--benchmark-only",
                     "--benchmark-json=data_reweighted_dynamics_scaling_time_horizon.json",
                     "--benchmark-min-rounds=2"])

        df = benchmark_utils._load_results('data_reweighted_dynamics_scaling_time_horizon.json')
        benchmark_utils._plot_results(df, "plot_reweighted_dynamics_scaling_time_horizon.pdf",
                                      r"$\text{Scaling benchmark for computation of reweighted dynamics}~P_W$")


    if benchmarks == "all" or benchmarks == "test_parameterized_dynamics_fits_scaling_time_horizon":
        print("Running benchmark for parameterized dynamics fits scaling with time horizon...")

        pytest.main([__file__,
                     "-k", "test_parameterized_dynamics_fits_scaling_time_horizon",
                     "--benchmark-enable", "--benchmark-only",
                     "--benchmark-json=data_parameterized_dynamics_fits_scaling_time_horizon.json"])

        df = benchmark_utils._load_results('data_parameterized_dynamics_fits_scaling_time_horizon.json')
        config_benchmark = benchmark_utils._get_config_benchmark("config_parameterized_dynamics_fits_scaling_time_horizon.json5")
        no_qubits = config_benchmark["no_qubits_list"][0]
        no_layers = config_benchmark["no_layers_list"][0]
        pqc_str = r"~\text{for " + f"{no_qubits} qubits, {no_layers} layers" + r"}$"
        benchmark_utils._plot_results(df, "plot_parameterized_dynamics_fits_scaling_time_horizon.pdf",
                                      r"$\text{Scaling benchmark for computation of fits of parameterized dynamics}~P_\theta~\text{to}~P_W"
                                      + pqc_str)
