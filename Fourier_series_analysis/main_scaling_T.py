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

from pathlib import Path
from logging_config import get_logger
from policy_evaluation_and_plots import PolicyEvaluation, plot_xy_vs_T
from utilities import load_and_restore_obj


logger = get_logger("main_scaling_T.py")


def main_scaling_T() -> None:
    logger.info("Starting main script.")
    # config with parameters of computations (see comments in class Config or in JSON5 file for their meaning)

    time_horizons = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    evaluation_reweighted_dynamics_dict = {}
    evaluation_fits_dict = {}

    for T in time_horizons:
        path_computations = Path(__file__).parent / f"results/config_scaling_T_{T}/computations"

        file_name = path_computations / f"evaluation_reweighted_dynamics.npz"
        evaluation_reweighted_dynamics = \
            load_and_restore_obj(PolicyEvaluation, file_name, None, check_params_consistency=False)
        evaluation_reweighted_dynamics_dict[f"T: {T}"] = evaluation_reweighted_dynamics

        file_name = path_computations / f"evaluation_fits_qubits_1_layers_3_leastsq_fitting_parameters_Fourier_coefficients.npz"
        evaluation_fits = \
            load_and_restore_obj(PolicyEvaluation, file_name, None, check_params_consistency=False)
        evaluation_fits_dict[f"T: {T}"] = evaluation_fits


    logger.info(f"Generation of overview plot.")
    # initialize lists for plotting
    min_KL_1_qubit_list = \
        [evaluation_fits_dict[f"T: {T}"].min_Kullback_Leibler_divergence_estimate
         for T in time_horizons]

    mean_KL_1_qubit_list = \
        [evaluation_fits_dict[f"T: {T}"].mean_Kullback_Leibler_divergence_estimate
         for T in time_horizons]

    std_KL_1_qubit_list = \
        [evaluation_fits_dict[f"T: {T}"].std_Kullback_Leibler_divergence_estimate
         for T in time_horizons]

    min_diff_prob_rare_trajectory_1_qubit_list = \
        [(evaluation_reweighted_dynamics_dict[f"T: {T}"].prob_rare_trajectory
          - evaluation_fits_dict[f"T: {T}"].prob_rare_trajectory_selected)
         for T in time_horizons]

    # plot results
    plot_xy_vs_T(time_horizons, "$D(P_{\\theta}\Vert P_W)$",
                 "$\Delta P(x_T = 0)$",
                 min_KL_1_qubit_list, mean_KL_1_qubit_list, std_KL_1_qubit_list, min_diff_prob_rare_trajectory_1_qubit_list,
                 save_fig_as=Path(__file__).parent / "results" / "plot_Fourier_series_fits_scaling_T.pdf",
                 show_plot=True)


    logger.info("Main script finished.")
    return


if __name__ == "__main__":
    main_scaling_T()

