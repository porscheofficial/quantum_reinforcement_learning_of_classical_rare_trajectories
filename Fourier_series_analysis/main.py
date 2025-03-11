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


import os
import shutil
import numpy as np
from logging_config import get_logger
from policy_evaluation_and_plots import PolicyEvaluation, plot_as_heatmap, convert_dict_to_data_frame, \
    plot_xy_vs_no_layers, plot_Fourier_coeffs
from reweighted_dynamics import ReweightedDynamics
from Fourier_series_analysis_and_fits import ParameterizedDynamicsFits, FourierSeriesAnalysis
from utilities import load_or_compute_obj, import_params_from_json5, convert_to_and_save_latex_string
from value_functions import ValueFunction


logger = get_logger("main.py")


def main(param_file_name: str = "config_publication.json5") -> None:
    logger.info("Starting main script.")

    logger.info(f"1. Creating folders and loading parameters of computations from {param_file_name}.")
    path_script = os.path.dirname(os.path.abspath(__file__))  # get directory of current script
    param_folder_name = param_file_name.split(".")[0]

    path_computations = os.path.join(path_script, f"results/{param_folder_name}/computations")
    os.makedirs(path_computations, exist_ok=True)

    path_plots = os.path.join(path_script, f"results/{param_folder_name}/plots")
    os.makedirs(path_plots, exist_ok=True)

    path_config = os.path.join(path_script, f"results/{param_folder_name}")
    shutil.copy(param_file_name, path_config)

    params = import_params_from_json5(param_file_name)

    # parameters of computations (see comments in CSV file for their meaning)
    T = params["T"]
    s = params["s"]
    x_T = params["x_T"]
    prob_step_up = params["prob_step_up"]
    no_qubits_list = params["no_qubits_list"]
    no_samples_variational_params = params["no_samples_variational_params"]
    no_layers_list = params["no_layers_list"]
    no_fits = params["no_fits"]
    fitting_parameters = params["fitting_parameters"]
    max_optimization_steps = params["max_optimization_steps"]
    cost_func_type = params["cost_func_type"]
    no_trajectories_cost_func = params["no_trajectories_cost_func"]
    no_trajectories_policy_evaluation = params["no_trajectories_policy_evaluation"]
    policy_selection_criterion = params["policy_selection_criterion"]
    recompute_stored = params["recompute_stored"]


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
                                    f"_samples_{no_samples_variational_params}.pdf",
                                    second_coeffs_samples=numeric_Fourier_series_analysis_2_layers.coeffs_samples_array)


    logger.info(f"7. Fitting in terms of Fourier coefficients for:")
    parameterized_dynamics_fits_dict = {}

    for no_qubits in no_qubits_list:
        for no_layers in no_layers_list:
            if fitting_parameters == "Fourier_coefficients" and no_qubits == 2 and no_layers > 1:
                continue  # for no_layers > 1, fitting in terms of Fourier coefficients is the same for 1 and 2 qubits

            logger.info(f"qubits: {no_qubits}, data-uploading layers: {no_layers}")
            parameterized_dynamics_fits = \
                load_or_compute_obj(ParameterizedDynamicsFits,
                                    lambda: ParameterizedDynamicsFits(reweighted_dynamics.reweighted_dynamics_P_W,
                                                                      no_qubits, no_layers, no_fits, fitting_parameters,
                                                                      cost_func_type,
                                                                      no_trajectories_cost_func=no_trajectories_cost_func,
                                                                      max_optimization_steps=max_optimization_steps,
                                                                      T=T, s=s, x_T=x_T, prob_step_up=prob_step_up,
                                                                      optimal_average_return=np.log(reweighted_dynamics.partition_function_Z),
                                                                      compute_in_parallel=True),
                                    f"{path_computations}/fits_qubits_{no_qubits}_layers_{no_layers}_"
                                    f"{cost_func_type}_fitting_parameters_{fitting_parameters}.npz", params,
                                    recompute=recompute_stored)

            parameterized_dynamics_fits_dict[f"(qubits: {no_qubits}, layers: {no_layers})"] = \
                parameterized_dynamics_fits


    logger.info(f"8. Evaluation of fitted policies.")
    evaluation_fits_dict = {}

    for no_qubits in no_qubits_list:
        for no_layers in no_layers_list:
            if no_qubits == 2 and no_layers > 1:
                continue

            policies_array = \
                parameterized_dynamics_fits_dict[f"(qubits: {no_qubits}, layers: {no_layers})"].fitted_policies_array

            evaluation_fits = \
                load_or_compute_obj(PolicyEvaluation,
                                    lambda: PolicyEvaluation(T, s, x_T, prob_step_up,
                                                             no_trajectories_policy_evaluation,
                                                             policies_array=policies_array,
                                                             reweighted_dynamics=reweighted_dynamics,
                                                             policy_selection_criterion=policy_selection_criterion),
                                    f"{path_computations}/evaluation_fits_qubits_{no_qubits}_"
                                    f"layers_{no_layers}_{cost_func_type}_fitting_parameters_{fitting_parameters}.npz",
                                    params, recompute=recompute_stored)

            evaluation_fits_dict[f"(qubits: {no_qubits}, layers: {no_layers})"] = evaluation_fits

            plot_data = convert_dict_to_data_frame({"qubits": no_qubits, "layers": no_layers, "no_fits": no_fits,
                                                    "fitting_parameters": fitting_parameters,
                                                    "cost_func_type": cost_func_type,
                                                    "no_trajectories_cost_func": no_trajectories_cost_func,
                                                    "max_optimization_steps": max_optimization_steps}
                                                   | evaluation_fits.__dict__)
            plot_as_heatmap(policies_array[evaluation_fits.index_selected_policy], "$P_{\\theta}(x - 1 | x, t)$",
                            save_fig_as=f"{path_plots}/selected_fit_qubits_{no_qubits}_layers_{no_layers}_"
                                        f"{cost_func_type}_fitting_parameters_{fitting_parameters}.pdf",
                            plot_complement=True, plot_data=plot_data,
                            plot_mask=np.isnan(reweighted_dynamics.reweighted_dynamics_P_W), value_limits=(0., 1.))


    logger.info(f"9. Generation of overview plot.")
    # initialize lists for plotting
    min_KL_1_qubit_list = \
        [evaluation_fits_dict[f"(qubits: 1, layers: {no_layers})"].min_Kullback_Leibler_divergence_estimate
         for no_layers in no_layers_list]
    min_KL_2_qubits = evaluation_fits_dict[f"(qubits: 2, layers: 1)"].min_Kullback_Leibler_divergence_estimate
    mean_KL_1_qubit_list = \
        [evaluation_fits_dict[f"(qubits: 1, layers: {no_layers})"].mean_Kullback_Leibler_divergence_estimate
         for no_layers in no_layers_list]
    mean_KL_2_qubits = evaluation_fits_dict[f"(qubits: 2, layers: 1)"].mean_Kullback_Leibler_divergence_estimate
    std_KL_1_qubit_list = \
        [evaluation_fits_dict[f"(qubits: 1, layers: {no_layers})"].std_Kullback_Leibler_divergence_estimate
         for no_layers in no_layers_list]
    std_KL_2_qubits = evaluation_fits_dict[f"(qubits: 2, layers: 1)"].std_Kullback_Leibler_divergence_estimate
    min_diff_prob_rare_trajectory_1_qubit_list = \
        [(evaluation_reweighted_dynamics.prob_rare_trajectory
          - evaluation_fits_dict[f"(qubits: 1, layers: {no_layers})"].prob_rare_trajectory_selected)
         for no_layers in no_layers_list]
    min_diff_prob_rare_trajectory_2_qubits = \
        (evaluation_reweighted_dynamics.prob_rare_trajectory
         - evaluation_fits_dict[f"(qubits: 2, layers: 1)"].prob_rare_trajectory_selected)

    # plot results
    plot_xy_vs_no_layers(no_layers_list, "$D(P_{\\theta}\Vert P_W)$",
                         "$\Delta P(x_T = 0)$", "min",
                         min_KL_1_qubit_list, min_KL_2_qubits, mean_KL_1_qubit_list, mean_KL_2_qubits,
                         std_KL_1_qubit_list, std_KL_2_qubits,
                         min_diff_prob_rare_trajectory_1_qubit_list, min_diff_prob_rare_trajectory_2_qubits,
                         save_fig_as=f"{path_plots}/plot_table_results_Fourier_series_fits.pdf")


    logger.info("Main script finished.")
    return


if __name__ == "__main__":
    main()

