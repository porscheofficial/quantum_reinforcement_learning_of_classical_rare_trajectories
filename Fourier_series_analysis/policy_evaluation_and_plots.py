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


import warnings
from audioop import reverse

import numpy as np
import matplotlib as mpl
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
from logging_config import get_logger
from reweighted_dynamics import ReweightedDynamics
from utilities import ConsistentParametersClass
from value_functions import ValueFunction


# global plot settings
global_size = 18
font_size = 18

mpl.rcParams['font.serif'] = 'cmr10'  # alternative: 'Times New Roman'
mpl.rcParams['font.sans-serif'] = 'cmr10'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams["axes.labelsize"] = global_size
mpl.rcParams["axes.titlesize"] = font_size
mpl.rcParams["font.size"] = font_size
mpl.rcParams["xtick.labelsize"] = global_size
mpl.rcParams["ytick.labelsize"] = global_size
plt.rcParams.update({"font.size": font_size})
plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['axes.formatter.use_mathtext'] = True  # use mathtext for axis formatting


# define color-blind-friendly palette
colors = ['#0072B2', '#E97B33', '#009E73', '#CC79A7', '#F0E442', '#56B4E9']


logger = get_logger("policy_evaluation_and_plots.py")


def convert_dict_to_data_frame(data: dict) -> pd.DataFrame:
    """
    Convert dictionary to pandas.DataFrame with columns "parameter" and "value" and parameters ordered alphabetically.

    Parameters:
        data: dictionary containing data

    Returns:
        df: DataFrame with columns "parameter", "value"
    """
    data_list = []

    for key, value in data.items():
        if not isinstance(value, np.ndarray):
            if isinstance(value, float):
                value_str = f"{value:.3e}"

            else:
                value_str = str(value)

            data_list.append([key.replace("_", "$\_$"), value_str])

    return pd.DataFrame(data_list, columns=["parameter", "value"])


def plot_as_heatmap(two_dim_array: np.ndarray, colorbar_label: str, title="", save_fig_as="",
                    plot_mask: np.ndarray = None, plot_complement=False, plot_data: pd.DataFrame = None,
                    value_limits: tuple[float, float] = None) -> None:
    """
    Plot 2D array as function of t and x in form of a heatmap and save it as PDF.

    Parameters:
        two_dim_array: array to be plotted
        colorbar_label: label of color bar
        title: if not "", set title of plot
        plot_mask: mask for plotting (if None, plot all values of two_dim_array)
                   (e.g., for consistency with plot of reweighted dynamics P_W,
                   plot_mask = np.isnan(reweighted_dynamics.reweighted_dynamics_P_W)
                   with reweighted_dynamics: ReweightedDynamics)
        plot_complement: if True, plot 1 - two_dim_array
        save_fig_as: if not "", save plot as PDF with name save_fig_as
        plot_data: if not None, plot table of data associated to plot as a second page in PDF file
        value_limits: if not None, set limits for displayed values in two_dim_array

    Returns:
        None
    """

    assert len(two_dim_array.shape) == 2, "two_dim_array must be 2-dimensional"

    # initialization
    T = len(two_dim_array)

    # prepare two_dim_array for plotting with imshow
    if plot_mask is not None:
        two_dim_array = np.where(plot_mask, np.nan, two_dim_array)

    two_dim_array = np.swapaxes(two_dim_array, 0, 1)
    # now indices x,t

    if plot_complement:
        two_dim_array = 1 - two_dim_array

    # plot as heatmap
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    two_dim_array = np.real(two_dim_array)

    if value_limits is not None:
        im = ax.imshow(two_dim_array, cmap='viridis', vmin=value_limits[0], vmax=value_limits[1])
    else:
        im = ax.imshow(two_dim_array, cmap='viridis')

    # set color bar, title, labels, ticks, and limits
    fig.colorbar(im, cax=cax, orientation='vertical', label=colorbar_label)

    if title != "":
        ax.set_title(title)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")

    ax.set_xticks(np.array([0, 1 * T // 4, 2 * T // 4, 3 * T // 4, T]),
                  labels=[str(0), str(1 * T // 4), str(2 * T // 4), str(3 * T // 4), str(T)])

    ax.set_ylim([1 / 2 * T - 2, 3 / 2 * T])
    ax.set_yticks(np.array([2 * T // 4 - 1, 3 * T // 4 - 1, T - 1, 5 * T // 4 - 1, 6 * T // 4 - 1]),
                  labels=[str(-2 * T // 4), str(-1 * T // 4), str(0), str(1 * T // 4), str(2 * T // 4)])

    # save plot
    #if save_fig_as != "":
    #    fig.savefig(save_fig_as, bbox_inches="tight")

    with PdfPages(save_fig_as) as pdf:
        # add plot
        if save_fig_as != "":
            pdf.savefig(bbox_inches="tight")

        plt.show()
        plt.close()

        # add table of data associated to plot as a second page
        if plot_data is not None and save_fig_as != "":
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.axis('tight')
            ax.axis('off')
            _ = ax.table(cellText=plot_data.values, cellLoc="left", colLabels=plot_data.columns, colLoc="left",
                         loc='center', fontsize=6)
            pdf.savefig(fig, bbox_inches="tight")  # save table
            plt.close()


def plot_Fourier_coeffs(no_layers: int, coeffs_samples: np.ndarray, save_fig_as: str,
                        second_coeffs_samples: np.ndarray = None) -> None:
    """
    Visualize Fourier coefficients as scatter plot.
    This function consists of code adjusted from the PennyLane demo by Schuld and Meyer
    (https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series/#part-iii-sampling-fourier-coefficients).

    Parameters:
        no_layers: number of layers
        coeffs_samples: coeffs_samples[j, m, n] contains the Fourier coefficient c_{m - no_layers, n} for frequency
                        n_x = m - no_layers and n_t = n of the j-th sample
        save_fig_as: name of PDF file to save plot
        second_coeffs_samples: if not None, second set of Fourier coefficients to be plotted

    Returns:
        None
    """

    # initialization
    _, no_x, no_t = coeffs_samples.shape

    coeffs_real = np.real(coeffs_samples)
    coeffs_imag = np.imag(coeffs_samples)

    if second_coeffs_samples is not None:
        _, no_x_2, no_t_2 = second_coeffs_samples.shape

        assert no_x == no_t_2, ("coeffs_samples and second_coeffs_samples do not have consistent shapes for current "
                                "version of plotting both")

        coeffs_real_2 = np.real(second_coeffs_samples)
        coeffs_imag_2 = np.imag(second_coeffs_samples)

    # plot Fourier coefficients
    if no_layers == 1:
        edge_color = colors[0]

    else:
        edge_color = colors[1]

    if second_coeffs_samples is None:
        fig, ax = plt.subplots(no_x, no_t, figsize=(2 * no_t, 2 * no_x), squeeze=False)

    else:
        fig, ax = plt.subplots(no_x, no_t + no_x_2, figsize=(2 * (no_t + no_x_2), 2 * no_x),
                               squeeze=False)

    for m in range(no_x):
        for n in range(no_t):
            ax[m, n].set_title("$c_{" + str(m - no_layers) + str(n) + "}$")
            ax[m, n].scatter(coeffs_real[:, m, n], coeffs_imag[:, m, n], s=20,
                             facecolor='white', edgecolor=edge_color)
            ax[m, n].set_aspect("equal")
            ax[m, n].set_ylim(-1, 1)
            ax[m, n].set_xlim(-1, 1)

    if second_coeffs_samples is not None:
        for m in range(no_x_2):
            for n in range(no_t_2):
                ax[n, no_t + m].set_title("$c_{" + str(m - no_layers) + str(n) + "}$")
                ax[n, no_t + m].scatter(coeffs_real_2[:, m, n], coeffs_imag_2[:, m, n], s=20,
                                 facecolor='white', edgecolor=colors[1])
                ax[n, no_t + m].set_aspect("equal")
                ax[n, no_t + m].set_ylim(-1, 1)
                ax[n, no_t + m].set_xlim(-1, 1)

    # set labels
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('real part')
    plt.ylabel('imaginary part')

    plt.tight_layout(pad=0.5)

    # save plot
    fig.savefig(save_fig_as, bbox_inches="tight")

    plt.show()
    plt.close()


def plot_xy_vs_no_layers(no_layers_list: list, quantity_x_label: str, quantity_y_label: str, opt: str,
                         opt_x_1_qubit_list: list, opt_x_2_qubits: float,
                         mean_x_1_qubit_list: list, mean_x_2_qubits: float,
                         std_x_1_qubit_list: list, std_x_2_qubits: float,
                         opt_y_1_qubit_list: list, opt_y_2_qubits: float,
                         save_fig_as="plot_table_results_Fourier_series_fits.pdf") -> None:
    """
    Create plot of quantities x and y for fitted parameterized dynamics of parameterized quantum circuits (PQCs)
    with 1 and 2 qubits vs. #data-uploading layers of PQCs.
    Plot includes optimal and mean values with standard deviations shown as error bars.

    Parameters:
        no_layers_list: list of #layers of PQCs
        quantity_x_label: label of quantity x
        quantity_y_label: label of quantity y
        opt: optimal values of x given by "max" or "min"
        opt_x_1_qubit_list: list of optimal values of x for 1-qubit PQCs
        opt_x_2_qubits: optimal value of x for 2-qubit PQCs with 1 layer
        mean_x_1_qubit_list: list of mean values of x for 1-qubit PQCs
        mean_x_2_qubits: mean value of x for 2-qubit PQCs with 1 layer
        std_x_1_qubit_list: list of standard deviations of x for 1-qubit PQCs
        std_x_2_qubits: standard deviation of x for 2-qubit PQCs with 1 layer
        opt_y_1_qubit_list: list of optimal values of y for 1-qubit PQCs
        opt_y_2_qubits: optimal value of y for 2-qubit PQCs with 1 layer
        save_fig_as: name of PDF file to save plot

    Returns:
        None
    """

    assert len(no_layers_list) == len(opt_x_1_qubit_list) == len(mean_x_1_qubit_list) == len(std_x_1_qubit_list), \
        ("Length of lists no_layers_list, opt_x_1_qubit_list, mean_x_1_qubit_list, and std_x_1_qubit_list must be "
         "equal.")

    fig, axes_1 = plt.subplots(1, 3, width_ratios=[3, 1, 1])
    plt.subplots_adjust(wspace=0.15)  # adjusts width between subplots

    # plot quantity x with error bars
    color = colors[0]

    for ax in axes_1:
        ax.errorbar(no_layers_list, opt_x_1_qubit_list, fmt='D', color=color)
        ax.errorbar(no_layers_list, mean_x_1_qubit_list, yerr=std_x_1_qubit_list, fmt='o', color=color)

        ax.errorbar(1, opt_x_2_qubits, fmt='D', mec=color, mfc='none')
        ax.errorbar(1, mean_x_2_qubits, yerr=std_x_2_qubits, fmt='o', mec=color, mfc='none')

    # create second vertical axis sharing same horizontal axis
    axes_2 = [ax.twinx() for ax in axes_1]

    # plot quantity y
    color = colors[1]

    for ax in axes_2:
        ax.errorbar(no_layers_list, opt_y_1_qubit_list, fmt='s', color=color)
        ax.errorbar(1, opt_y_2_qubits, fmt='s', mec=color, mfc='none')

    # adjust plots
    for axes in (axes_1, axes_2):
        axes[0].spines['right'].set_visible(False)
        axes[1].spines['left'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[2].spines['left'].set_visible(False)

    axes_1[0].set_xlim(0., 6.)
    axes_1[1].set_xlim(9., 11.)
    axes_1[2].set_xlim(14., 16.)

    all_opt_x_values = opt_x_1_qubit_list + [opt_x_2_qubits]
    all_mean_x_values = mean_x_1_qubit_list + [mean_x_2_qubits]
    all_std_x_values = std_x_1_qubit_list + [std_x_2_qubits]

    all_x_values = all_opt_x_values + [all_mean_x_values[i] + all_std_x_values[i]
                                       for i in range(len(all_opt_x_values))]

    if opt == "max":
        ylim_x = 1.1 * np.min(all_x_values)

    elif opt == "min":
        ylim_x = 1.1 * np.max(all_x_values)

    else:
        raise ValueError(f"Invalid opt {opt}.")

    for ax in axes_1:
        ax.set_ylim(0.0, ylim_x)

    for ax in axes_2:
        ax.set_ylim(0.0, 1.0)

    axes_1[0].set_xticks([1, 2, 3, 4, 5])
    axes_1[0].set_xticklabels([1, 2, 3, 4, 5])
    axes_1[1].set_xticks([10])
    axes_1[1].set_xticklabels([10])
    axes_1[2].set_xticks([15])
    axes_1[2].set_xticklabels([15])

    tick_step = ylim_x / 5.
    axes_1[0].set_yticks([0., tick_step, 2 * tick_step, 3 * tick_step, 4 * tick_step, 5 * tick_step])
    axes_1[0].set_yticklabels([f"{x:.2f}" for x
                               in [0., tick_step, 2 * tick_step, 3 * tick_step, 4 * tick_step, 5 * tick_step]])

    color = colors[0]
    axes_1[0].set_ylabel(quantity_x_label, color=color)
    axes_1[0].tick_params(axis='y', labelcolor=color, which='both',
                          labelleft=True, left=True, labelright=False, right=False)
    axes_1[1].tick_params(axis='y', labelleft=False, left=False, labelright=False, right=False)
    axes_1[2].tick_params(axis='y', labelleft=False, left=False, labelright=False, right=False)

    color = colors[1]
    axes_2[2].set_ylabel(quantity_y_label, color=color)
    axes_2[0].tick_params(axis='y', labelleft=False, left=False, labelright=False, right=False)
    axes_2[1].tick_params(axis='y', labelleft=False, left=False, labelright=False, right=False)
    axes_2[2].tick_params(axis='y', labelcolor=color, which='both',
                          labelleft=False, left=False, labelright=True, right=True)

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('# data-uploading layers')

    fig.tight_layout()

    fig.savefig(save_fig_as, bbox_inches="tight")

    plt.show()
    plt.close()


class PolicyEvaluation(ConsistentParametersClass):
    """
    Class to evaluate various properties of policies such as estimates for probabilities to generate rare trajectories,
    for average return values, for Kullback-Leibler divergences and mean squared errors to reweighted dynamics,
    and to compute statistics of these quantities for all considered policies.
    """
    def __init__(self, T: int, s: float, x_T: int, prob_step_up: float, no_trajectories_policy_evaluation: int,
                 policies_array: np.ndarray = None, reweighted_dynamics: ReweightedDynamics = None,
                 policy_selection_criterion: str = None):
        """
        Evaluate various properties of policies such as estimates for probabilities to generate rare trajectories,
        for average return values, for Kullback-Leibler divergences and mean squared errors to reweighted dynamics,
        and compute statistics of these quantities for all considered policies.

        Parameters:
            T: #time steps of random walk
            s: softening parameter
            x_T: required end point of rare trajectory
            prob_step_up: probability to go 1 step up
            no_trajectories_policy_evaluation: #trajectories to be generated
            policies_array: policy_array[n, t, x] contains probability of moving to x + 1 at time t according to
                          n-th policy
            reweighted_dynamics: reweighted dynamics to be compared with (if not None, Kullback-Leibler divergence
                                 estimate and mean squared error is computed)

        Returns:
            object of class PolicyEvaluation with estimates for probabilities to generate rare trajectories,
            for average return values, for Kullback-Leibler divergences and mean squared errors to reweighted dynamics,
            and statistics of these quantities for all considered policies
        """

        super().__init__()

        # asserts
        assert T > 0, "T > 0 required"
        assert s >= 0., "s >= 0. required"
        assert 0. <= prob_step_up <= 1., "0 <= prob_step_up <= 1 required"
        assert no_trajectories_policy_evaluation > 0, "no_trajectories > 0 required"

        # save inputs
        self.T = T
        self.s = s
        self.x_T = x_T
        self.prob_step_up = prob_step_up
        self.no_trajectories_policy_evaluation = no_trajectories_policy_evaluation

        if policies_array is not None:
            self.policies_array = policies_array
        else:
            self.policies_array = np.expand_dims(reweighted_dynamics.reweighted_dynamics_P_W, 0)

        if reweighted_dynamics is not None:
            self.reweighted_dynamics_P_W = reweighted_dynamics.reweighted_dynamics_P_W
            self.average_return_P_W = np.log(reweighted_dynamics.partition_function_Z)

        # generate trajectories from policy_array
        trajectories_x_array = self.calc_trajectories_x_array(self.policies_array, T, no_trajectories_policy_evaluation)

        # compute estimates for probabilities to generate rare trajectories
        self.prob_rare_trajectory = self.calc_prob_rare_trajectory(trajectories_x_array, x_T)

        if policies_array is not None:
            self.max_prob_rare_trajectory = np.max(self.prob_rare_trajectory)
            self.mean_prob_rare_trajectory = np.mean(self.prob_rare_trajectory)
            self.std_prob_rare_trajectory = np.std(self.prob_rare_trajectory)

        else:
            self.prob_rare_trajectory = self.prob_rare_trajectory.item()

        # compute estimates for average return values
        self.return_values_array = self.calc_return_values(trajectories_x_array, self.policies_array, s, x_T,
                                                           prob_step_up)
        self.average_return_estimate = np.mean(self.return_values_array, axis=-1)

        if policies_array is not None:
            self.max_average_return_estimate = np.max(self.average_return_estimate)
            self.mean_average_return_estimate = np.mean(self.average_return_estimate)
            self.std_average_return_estimate = np.std(self.average_return_estimate)

        else:
            self.average_return_estimate = self.average_return_estimate.item()

        # compute estimates for Kullback-Leibler divergences
        if reweighted_dynamics is not None:
            self.Kullback_Leibler_divergence_estimate = - self.average_return_estimate \
                                                        + self.average_return_P_W
            # np.log(self.partition_function_Z) is the average return value of the reweighted dynamics

            if policies_array is not None:
                self.min_Kullback_Leibler_divergence_estimate = np.min(self.Kullback_Leibler_divergence_estimate)
                self.mean_Kullback_Leibler_divergence_estimate = np.mean(self.Kullback_Leibler_divergence_estimate)
                self.std_Kullback_Leibler_divergence_estimate = np.std(self.Kullback_Leibler_divergence_estimate)

        # perform consistency check
        if policies_array is None:
            deviation_from_exact_average_return = (np.abs(self.Kullback_Leibler_divergence_estimate)
                                                   / np.abs(self.average_return_P_W))

            if deviation_from_exact_average_return > 1e-2:
                warning_message = (f"Deviation from exact average return value for reweighted dynamics is "
                                   f"{deviation_from_exact_average_return * 100}%. "
                                   f"Thus it is advised to use a higher value for 'no_trajectories_policy_evaluation' "
                                   f"than the current value {no_trajectories_policy_evaluation}.")

                warnings.warn(warning_message)
                logger.warning(warning_message)

        # compute mean squared errors to reweighted dynamics
        if reweighted_dynamics is not None and policies_array is not None:
            self.mean_squared_error_to_P_W = np.nanmean((policies_array - self.reweighted_dynamics_P_W) ** 2,
                                                        axis=(-2, -1))

            self.min_mean_squared_error_to_P_W = np.min(self.mean_squared_error_to_P_W)
            self.mean_mean_squared_error_to_P_W = np.mean(self.mean_squared_error_to_P_W)
            self.std_mean_squared_error_to_P_W = np.std(self.mean_squared_error_to_P_W)

        # select policy according to criterion
        if policy_selection_criterion is not None:
            match policy_selection_criterion:
                case "max_prob_rare_trajectory":
                    self.index_selected_policy = np.argmax(self.prob_rare_trajectory)

                case "max_avg_return":
                    self.index_selected_policy = np.argmax(self.average_return_estimate)

                case "min_KL_divergence":
                    self.index_selected_policy = np.argmin(self.Kullback_Leibler_divergence_estimate)

                case "min_MSE":
                    self.index_selected_policy = np.argmin(self.mean_squared_error_to_P_W)

            self.prob_rare_trajectory_selected = self.prob_rare_trajectory[self.index_selected_policy]
            self.average_return_estimate_selected = self.average_return_estimate[self.index_selected_policy]

            if reweighted_dynamics is not None and policies_array is not None:
                self.Kullback_Leibler_divergence_estimate_selected = \
                    self.Kullback_Leibler_divergence_estimate[self.index_selected_policy]

            if reweighted_dynamics is not None and policies_array is not None:
                self.mean_squared_error_to_P_W_selected = self.mean_squared_error_to_P_W[self.index_selected_policy]


    @property
    def all_init_params_dict(self):
        return {"T": self.T, "s": self.s, "x_T": self.x_T, "prob_step_up": self.prob_step_up,
                "no_trajectories_policy_evaluation": self.no_trajectories_policy_evaluation}


    @staticmethod
    def calc_trajectories_x_array(policies_array: np.ndarray, T: int, no_trajectories: int) -> np.ndarray:
        """
        Compute array of trajectories (x_0, x_1, ..., x_T) of length T + 1 with x_0 = 0 according to policies.

        Parameters:
            policies_array: policies_array[m, t, x + T - 1] contains the probability of moving to x + 1 at time t
                            according to the m-th policy
            T: length of trajectories
            no_trajectories: number of trajectories

        Returns:
            trajectories_x_array: trajectories_x_array[m, n, :] contains the n-th trajectory generated by the
                                  m-th policy
        """

        # asserts
        assert T > 0, "T > 0 required"
        assert no_trajectories > 0, "no_trajectories > 0 required"

        # initialization
        no_policies = len(policies_array)
        trajectories_x_array = np.zeros((no_policies, no_trajectories, T + 1), dtype=int)
        rng = np.random.default_rng()  # random number generator

        # compute array of trajectory positions
        for t in range(1, T + 1):
            delta_x = np.array([[rng.choice([+1, -1],
                                            p=[policies_array[m, t - 1, trajectories_x_array[m, n, t - 1] + T - 1],
                                               1 - policies_array[m, t - 1, trajectories_x_array[m, n, t - 1] + T - 1]])
                                 # + T - 1 due to the way in which policies_array is defined
                                 for n in range(no_trajectories)]
                                for m in range(no_policies)])
            trajectories_x_array[:, :, t] = trajectories_x_array[:, :, t - 1] + delta_x

        return trajectories_x_array


    def calc_prob_rare_trajectory(self, trajectories_x_array: np.ndarray, x_T: float) -> np.ndarray:
        """
        Compute array of empirical estimate for probabilities of generating rare trajectories.

        Parameters:
            trajectories_x_array: trajectories_x_array[m, n, :] contains the n-th trajectory generated by the
                                  m-th policy
            x_T: required end point of rare trajectory

        Returns:
            prob_rare_trajectory: prob_rare_trajectory[m] contains the empirical estimate for the probability of
                                  generating a rare trajectory according to the m-th policy
        """

        no_trajectories = np.shape(trajectories_x_array)[1]

        no_rare_trajectories = np.sum(trajectories_x_array[:, :, -1] == x_T, axis=-1)
        # no. of trajectories with end point == x_T

        return no_rare_trajectories / no_trajectories


    @staticmethod
    def calc_return_values(trajectories_x_array: np.ndarray, policies_array: np.ndarray, s: float, x_T: int,
                           prob_step_up: float) -> np.ndarray:
        """
        Compute array of return values for trajectories.

        Parameters:
            trajectories_x_array: trajectories_x_array[m, n, :] contains the n-th trajectory generated by the
                                  m-th policy
            policies_array: policies_array[m, t, x + T - 1] contains the probability of moving to x + 1 at time t
                            according to the m-th policy
            s: factor contained in the reward function
                (determining the importance of generating rare trajectories vs. staying close to the original dynamics
                of the random walk)

        Returns:
            return_values_array: return_values_list[m, n] contains the return value of the n-th trajectory generated by
                                 the m-th policy
        """

        # asserts
        assert s >= 0., "s >= 0. required"
        assert 0. <= prob_step_up <= 1., "0 <= prob_step_up <= 1 required"

        # initializations
        no_policies, no_trajectories, length = np.shape(trajectories_x_array)
        T = length - 1

        return_values_array = np.zeros((no_policies, no_trajectories))

        # compute return values for trajectories
        for t in range(1, T + 1):
            return_values_array += \
                np.array([[ValueFunction.calc_reward(trajectories_x_array[m, n, t] - trajectories_x_array[m, n, t - 1],
                                                     trajectories_x_array[m, n, t - 1], t, policies_array[m],
                                                     T, s, x_T, prob_step_up)
                           for n in range(no_trajectories)]
                          for m in range(no_policies)])

        return return_values_array

