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
import shutil
import numpy as np

from config_template import Config
from logging_config import get_logger
from policy_evaluation_and_plots import PolicyEvaluation, plot_as_heatmap, convert_dict_to_data_frame, \
    plot_xy_vs_no_layers, plot_Fourier_coeffs
from reweighted_dynamics import ReweightedDynamics
from Fourier_series_analysis_and_fits import ParameterizedDynamicsFits, FourierSeriesAnalysis
from utilities import get_file_names_with_version, load_or_compute_obj, convert_to_and_save_latex_string
from value_functions import ValueFunction


logger = get_logger("main.py")


def main(config_file_name: str = "config_publication.json5", create_plots=True) -> None:
    logger.info("Starting main script.")
    # config with parameters of computations (see comments in class Config or in JSON5 file for their meaning)
    config = Config.from_json5(config_file_name)

    params = config.__dict__
    T = config.T
    s = config.s
    x_T = config.x_T
    prob_step_up = config.prob_step_up
    no_qubits_list = config.no_qubits_list
    no_samples_variational_params = config.no_samples_variational_params
    no_layers_list = config.no_layers_list
    no_fits = config.no_fits
    fitting_parameters = config.fitting_parameters
    no_random_Fourier_features = config.no_random_Fourier_features
    no_choices_random_Fourier_features = config.no_choices_random_Fourier_features
    max_optimization_steps = config.max_optimization_steps
    cost_func_type = config.cost_func_type
    no_trajectories_cost_func = config.no_trajectories_cost_func
    no_trajectories_policy_evaluation = config.no_trajectories_policy_evaluation
    policy_selection_criterion = config.policy_selection_criterion
    recompute_stored = config.recompute_stored


    logger.info(f"1. Creating folders and loading parameters of computations from {config_file_name}.")
    path_script = os.path.dirname(os.path.abspath(__file__))  # get directory of current script
    config_folder_name = config_file_name.split(".")[0]

    path_computations = os.path.join(path_script, f"results/{config_folder_name}/computations")
    os.makedirs(path_computations, exist_ok=True)

    path_plots = os.path.join(path_script, f"results/{config_folder_name}/plots")
    os.makedirs(path_plots, exist_ok=True)

    path_config = os.path.join(path_script, f"results/{config_folder_name}")
    shutil.copy(config_file_name, path_config)


    logger.info(f"2. Computation of reweighted dynamics.")
    reweighted_dynamics = \
        load_or_compute_obj(ReweightedDynamics,
                            lambda: ReweightedDynamics(T, s, x_T, prob_step_up),
                            f"{path_computations}/reweighted_dynamics.npz", params,
                            recompute=recompute_stored)


    logger.info(f"3. Evaluation of original and reweighted dynamics.")
    original_dynamics_P = np.where(np.isnan(reweighted_dynamics.reweighted_dynamics_P_W), np.nan, prob_step_up)

    evaluation_original_dynamics = \
        load_or_compute_obj(PolicyEvaluation,
                            lambda: PolicyEvaluation(T, s, x_T, prob_step_up, no_trajectories_policy_evaluation,
                                                     policies_array=np.expand_dims(original_dynamics_P, axis=0),
                                                     reweighted_dynamics=reweighted_dynamics),
                            f"{path_computations}/evaluation_original_dynamics.npz", params,
                            recompute=recompute_stored)

    plot_data = convert_dict_to_data_frame(evaluation_original_dynamics.__dict__)
    plot_as_heatmap(original_dynamics_P, "$P(x - 1 | x, t)$",
                    save_fig_as=f"{path_plots}/P_to_go_1_step_down.pdf", plot_complement=True, plot_data=plot_data,
                    value_limits=(0., 1.))

    evaluation_reweighted_dynamics = \
        load_or_compute_obj(PolicyEvaluation,
                            lambda: PolicyEvaluation(T, s, x_T, prob_step_up, no_trajectories_policy_evaluation,
                                                     reweighted_dynamics=reweighted_dynamics),
                            f"{path_computations}/evaluation_reweighted_dynamics.npz", params,
                            recompute=recompute_stored)

    plot_data = convert_dict_to_data_frame(evaluation_reweighted_dynamics.__dict__)
    plot_as_heatmap(reweighted_dynamics.reweighted_dynamics_P_W, "$P_W(x - 1 | x, t)$",
                    save_fig_as=f"{path_plots}/P_W_to_go_1_step_down.pdf", plot_complement=True, plot_data=plot_data,
                    value_limits=(0., 1.))


    logger.info(f"4. Computation of value function for reweighted dynamics.")
    value_function_reweighted_dynamics = \
        load_or_compute_obj(ValueFunction,
                            lambda: ValueFunction(reweighted_dynamics.reweighted_dynamics_P_W, T, s, x_T, prob_step_up),
                            f"{path_computations}/value_function_reweighted_dynamics.npz", params,
                            recompute=recompute_stored)

    plot_as_heatmap(np.log10(-value_function_reweighted_dynamics.value_func_array[:-1]),
                    # [:1] to discard value function values V(x, T) == 0 for plotting
                    "log$_{10}(V_{P_W}(x, t))$", save_fig_as=f"{path_plots}/V_P_W.pdf")


    logger.info(f"5. Symbolic calculation of Fourier coefficients for:")
    for no_qubits in no_qubits_list:
        for no_layers in no_layers_list:
            if no_layers == 1:
                # for more data-uploading layers, the symbolic Fourier coefficients are not informative and
                # might be incorrect due to a known SymPy bug
                logger.info(f"qubits: {no_qubits}, data-uploading layers: {no_layers}")
                symbolic_Fourier_series_analysis = \
                    load_or_compute_obj(FourierSeriesAnalysis,
                                        lambda: FourierSeriesAnalysis(no_qubits, no_layers, "symbolic"),
                                        f"{path_computations}/symbolic_Fourier_series_analysis_"
                                        f"qubits_{no_qubits}_layers_{no_layers}.npz",
                                        params, recompute=recompute_stored, load=True)
                # use load=False to avoid loading object from file in case of memory issues

                if symbolic_Fourier_series_analysis is not None:
                    convert_to_and_save_latex_string(symbolic_Fourier_series_analysis.amp_phase_series,
                                                     f"{path_computations}/amp_phase_series_qubits_{no_qubits}_"
                                                     f"layers_{no_layers}.txt",
                                                     f"Fourier series in amplitude-phase form for "
                                                     f"#qubits: {no_qubits}, #data-uploading layers: {no_layers}")


    logger.info(f"6. Numerical computation of Fourier coefficients for:")
    numeric_Fourier_series_analysis_1_layer = None
    numeric_Fourier_series_analysis_2_layers = None

    for no_qubits in no_qubits_list:
        for no_layers in no_layers_list:
            if no_qubits == 2 and no_layers > 8:
                continue

            logger.info(f"qubits: {no_qubits}, data-uploading layers: {no_layers}")
            numeric_Fourier_series_analysis = \
                load_or_compute_obj(FourierSeriesAnalysis,
                                    lambda: FourierSeriesAnalysis(no_qubits, no_layers, "numeric",
                                                                  no_samples_variational_params, random_thetas=True),
                                    f"{path_computations}/numeric_Fourier_series_analysis_qubits_{no_qubits}_"
                                    f"layers_{no_layers}_samples_{no_samples_variational_params}.npz",
                                    params, recompute=recompute_stored, load=True)
            # use load=False to avoid loading object from file in case of memory issues

            if numeric_Fourier_series_analysis is not None:
                plot_Fourier_coeffs(no_layers, numeric_Fourier_series_analysis.coeffs_samples_array,
                                    f"{path_plots}/Fourier_coeffs_qubits_{no_qubits}_layers_{no_layers}"
                                    f"_samples_{no_samples_variational_params}.pdf")

            # produce plots in publication
            if no_layers == 1:
                numeric_Fourier_series_analysis_1_layer = numeric_Fourier_series_analysis
            if no_layers == 2:
                numeric_Fourier_series_analysis_2_layers = numeric_Fourier_series_analysis

            if numeric_Fourier_series_analysis_1_layer is not None and \
                    numeric_Fourier_series_analysis_2_layers is not None:
                plot_Fourier_coeffs(1, numeric_Fourier_series_analysis_1_layer.coeffs_samples_array,
                                    f"{path_plots}/Fourier_coeffs_qubits_{no_qubits}_layers_1_and_2"
                                    f"_samples_{no_samples_variational_params}.pdf", second_no_layers=2,
                                    second_coeffs_samples=numeric_Fourier_series_analysis_2_layers.coeffs_samples_array)


    logger.info(f"7. Fitting in terms of Fourier coefficients for:")
    parameterized_dynamics_fits_dict = {}

    for no_qubits in no_qubits_list:
        for no_layers in no_layers_list:
            if fitting_parameters in ("Fourier_coefficients", "random_Fourier_features") and no_qubits == 2 and no_layers > 1:
                continue  # for no_layers > 1, fitting in terms of Fourier coefficients is the same for 1 and 2 qubits

            if fitting_parameters in ("Fourier_coefficients", "variational_angles"):
                logger.info(f"qubits: {no_qubits}, data-uploading layers: {no_layers}")

                file_name_list = [f"{path_computations}/fits_qubits_{no_qubits}_layers_{no_layers}_{cost_func_type}_"
                                  f"fitting_parameters_{fitting_parameters}.npz"]
                end_version = 1

            if fitting_parameters == "random_Fourier_features":
                logger.info(f"qubits: {no_qubits}, data-uploading layers: {no_layers}, "
                            f"random_Fourier_features: {no_random_Fourier_features}")

                file_name = (f"{path_computations}/fits_qubits_{no_qubits}_layers_{no_layers}_{cost_func_type}_"
                             f"random_Fourier_features_{no_random_Fourier_features}.npz")

                file_name_list, end_version = get_file_names_with_version(file_name, no_choices_random_Fourier_features,
                                                                          path_computations)

            for file_name in file_name_list:
                parameterized_dynamics_fits = \
                    load_or_compute_obj(ParameterizedDynamicsFits,
                                        lambda: ParameterizedDynamicsFits(reweighted_dynamics.reweighted_dynamics_P_W,
                                                                          no_qubits, no_layers, no_fits, fitting_parameters,
                                                                          cost_func_type,
                                                                          no_trajectories_cost_func=no_trajectories_cost_func,
                                                                          max_optimization_steps=max_optimization_steps,
                                                                          no_random_Fourier_features=no_random_Fourier_features,
                                                                          T=T, s=s, x_T=x_T, prob_step_up=prob_step_up,
                                                                          optimal_average_return=np.log(reweighted_dynamics.partition_function_Z),
                                                                          compute_in_parallel=True),
                                        file_name, params, recompute=recompute_stored)

                if fitting_parameters in ("Fourier_coefficients", "variational_angles"):
                    parameterized_dynamics_fits_dict[(f"(qubits: {no_qubits}, layers: {no_layers}, version: 1")] = \
                        parameterized_dynamics_fits

                if fitting_parameters == "random_Fourier_features":
                    parameterized_dynamics_fits_dict[(f"(qubits: {no_qubits}, layers: {no_layers}, "
                                                      f"version: {file_name.split('_v')[1].split('.')[0]})")] = \
                        parameterized_dynamics_fits


    if not create_plots:
        logger.info("Skipping plots creation as requested.")
        logger.info("Main script finished.")
        return

    logger.info(f"8. Evaluation of fitted policies.")
    evaluation_fits_dict = {}

    for no_qubits in no_qubits_list:
        for no_layers in no_layers_list:
            if no_qubits == 2 and no_layers > 1:
                continue

            if fitting_parameters in ("Fourier_coefficients", "variational_angles"):
                file_name_list = [f"{path_computations}/evaluation_fits_qubits_{no_qubits}_layers_{no_layers}_{cost_func_type}_"
                                  f"fitting_parameters_{fitting_parameters}.npz"]

                plot_file_name_list = [f"{path_plots}/selected_fit_qubits_{no_qubits}_layers_{no_layers}_{cost_func_type}_"
                                       f"fitting_parameters_{fitting_parameters}.pdf"]

            if fitting_parameters == "random_Fourier_features":
                file_name = (f"{path_computations}/evaluation_fits_qubits_{no_qubits}_layers_{no_layers}_{cost_func_type}_"
                             f"random_Fourier_features_{no_random_Fourier_features}.npz")

                plot_file_name = (f"{path_plots}/selected_fit_qubits_{no_qubits}_layers_{no_layers}_{cost_func_type}_"
                                  f"random_Fourier_features_{no_random_Fourier_features}.pdf")

                file_name_list, _ = get_file_names_with_version(file_name, no_choices_random_Fourier_features,
                                                                path_computations)

                plot_file_name_list, _ = get_file_names_with_version(plot_file_name, no_choices_random_Fourier_features,
                                                                     path_plots)

            for n, file_name in enumerate(file_name_list):
                policies_array = \
                    parameterized_dynamics_fits_dict[
                        (f"(qubits: {no_qubits}, layers: {no_layers}, "
                         f"version: {file_name.split('_v')[1].split('.')[0] if fitting_parameters == 'random_Fourier_features' else 1})")
                    ].fitted_policies_array

                evaluation_fits = \
                    load_or_compute_obj(PolicyEvaluation,
                                        lambda: PolicyEvaluation(T, s, x_T, prob_step_up,
                                                                 no_trajectories_policy_evaluation,
                                                                 policies_array=policies_array,
                                                                 reweighted_dynamics=reweighted_dynamics,
                                                                 policy_selection_criterion=policy_selection_criterion),
                                        file_name, params, recompute=recompute_stored)


                if fitting_parameters in ("Fourier_coefficients", "variational_angles"):
                    evaluation_fits_dict[(f"(qubits: {no_qubits}, layers: {no_layers}, version: 1")] = evaluation_fits

                    plot_data = convert_dict_to_data_frame({"qubits": no_qubits, "layers": no_layers, "no_fits": no_fits,
                                                            "fitting_parameters": fitting_parameters,
                                                            "cost_func_type": cost_func_type,
                                                            "no_trajectories_cost_func": no_trajectories_cost_func,
                                                            "max_optimization_steps": max_optimization_steps}
                                                           | evaluation_fits.__dict__)

                if fitting_parameters == "random_Fourier_features":
                    evaluation_fits_dict[(f"(qubits: {no_qubits}, layers: {no_layers}, "
                                          f"version: {file_name.split('_v')[1].split('.')[0]})")] = evaluation_fits

                    plot_data = convert_dict_to_data_frame({"qubits": no_qubits, "layers": no_layers, "no_fits": no_fits,
                                                            "fitting_parameters": fitting_parameters,
                                                            "no_random_Fourier_features": no_random_Fourier_features,
                                                            "no_choices_random_Fourier_features": no_choices_random_Fourier_features,
                                                            "cost_func_type": cost_func_type,
                                                            "no_trajectories_cost_func": no_trajectories_cost_func,
                                                            "max_optimization_steps": max_optimization_steps}
                                                           | evaluation_fits.__dict__)

                plot_as_heatmap(policies_array[evaluation_fits.index_selected_policy], "$P_{\\theta}(x - 1 | x, t)$",
                                save_fig_as=plot_file_name_list[n], plot_complement=True, plot_data=plot_data,
                                plot_mask=np.isnan(reweighted_dynamics.reweighted_dynamics_P_W), value_limits=(0., 1.))


    # TODO: check whether the following code works correctly if fitting_parameters == "random_Fourier_features"
    logger.info(f"9. Generation of overview plot.")
    # initialize lists for plotting
    min_KL_1_qubit_list = \
        [[evaluation_fits_dict[f"(qubits: 1, layers: {no_layers}, version: {version_no})"].min_Kullback_Leibler_divergence_estimate
          for no_layers in no_layers_list]
         for version_no in range(1, end_version + 1)]
    try:
        min_KL_2_qubits_list = \
            [evaluation_fits_dict[f"(qubits: 2, layers: 1, version: {version_no})"].min_Kullback_Leibler_divergence_estimate
             for version_no in range(1, end_version + 1)]
    except KeyError:
        min_KL_2_qubits_list = [np.nan]

    mean_KL_1_qubit_list = \
        [[evaluation_fits_dict[f"(qubits: 1, layers: {no_layers}, version: {version_no})"].mean_Kullback_Leibler_divergence_estimate
          for no_layers in no_layers_list]
         for version_no in range(1, end_version + 1)]
    try:
        mean_KL_2_qubits_list = \
            [evaluation_fits_dict[f"(qubits: 2, layers: 1, version: {version_no})"].mean_Kullback_Leibler_divergence_estimate
             for version_no in range(1, end_version + 1)]
    except KeyError:
        mean_KL_2_qubits_list = [np.nan]

    std_KL_1_qubit_list = \
        [[evaluation_fits_dict[f"(qubits: 1, layers: {no_layers}, version: {version_no})"].std_Kullback_Leibler_divergence_estimate
          for no_layers in no_layers_list]
         for version_no in range(1, end_version + 1)]
    try:
        std_KL_2_qubits_list = \
            [evaluation_fits_dict[f"(qubits: 2, layers: 1, version: {version_no})"].std_Kullback_Leibler_divergence_estimate
             for version_no in range(1, end_version + 1)]
    except KeyError:
        std_KL_2_qubits_list = [np.nan]

    min_diff_prob_rare_trajectory_1_qubit_list = \
        [[(evaluation_reweighted_dynamics.prob_rare_trajectory
           - evaluation_fits_dict[f"(qubits: 1, layers: {no_layers}, version: {version_no})"].prob_rare_trajectory_selected)
          for no_layers in no_layers_list]
         for version_no in range(1, end_version + 1)]
    try:
        min_diff_prob_rare_trajectory_2_qubits_list = \
            [(evaluation_reweighted_dynamics.prob_rare_trajectory
              - evaluation_fits_dict[f"(qubits: 2, layers: 1, version: {version_no})"].prob_rare_trajectory_selected)
             for version_no in range(1, end_version + 1)]
    except KeyError:
        min_diff_prob_rare_trajectory_2_qubits_list = [np.nan]

    # plot results
    if fitting_parameters in ("Fourier_coefficients", "variational_angles"):
        file_name = (f"{path_plots}/plot_table_results_Fourier_series_fits.pdf")

    if fitting_parameters == "random_Fourier_features":
        file_name = (f"{path_plots}/plot_table_results_Fourier_series_fits_"
                     f"random_Fourier_features_{no_random_Fourier_features}.pdf")

    plot_xy_vs_no_layers(no_layers_list, "$D(P_{\\theta}\Vert P_W)$",
                         "$\Delta P(x_T = 0)$",
                         min_KL_1_qubit_list, min_KL_2_qubits_list, mean_KL_1_qubit_list, mean_KL_2_qubits_list,
                         std_KL_1_qubit_list, std_KL_2_qubits_list,
                         min_diff_prob_rare_trajectory_1_qubit_list, min_diff_prob_rare_trajectory_2_qubits_list,
                         save_fig_as=file_name)


    logger.info("Main script finished.")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Fourier series analysis with optional specification of config file.")
    parser.add_argument("--config_file_name", type=str, default="config_publication.json5",
                        help="Path to the config file (default: config_publication.json5)")
    args = parser.parse_args()
    main(args.config_file_name)

