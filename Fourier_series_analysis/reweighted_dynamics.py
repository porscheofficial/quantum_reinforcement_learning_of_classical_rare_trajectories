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


import numpy as np
from utilities import ConsistentParametersClass


class ReweightedDynamics(ConsistentParametersClass):
    """
    Class to compute reweighted dynamics and related quantities for rare trajectories of random walk with softened
    constraint.
    """
    def __init__(self, T: int, s: float, x_T: int, prob_step_up: float):
        """
        Compute reweighted dynamics and related quantities for rare trajectories of random walk with
        softened constraint.

        Parameters:
            T: #time steps of random walk
            s: softening parameter
            x_T: required end point of rare trajectory
            prob_step_up: probability to go 1 step up

        Returns:
            object of class ReweightedDynamics with reweighted dynamics P_W and related quantities
        """

        super().__init__()

        # asserts
        assert T > 0, "T > 0 required"
        assert s >= 0., "s >= 0. required"
        assert 0. <= prob_step_up <= 1., "0 <= prob_step_up <= 1 required"

        # save inputs
        self.T = T
        self.s = s
        self.x_T = x_T
        self.prob_step_up = prob_step_up

        # compute gauge transform necessary in chosen way to exactly compute reweighted dynamics
        self.gauge_transform_g = self.calc_gauge_transform_array(T, s, x_T=x_T, prob_step_up=prob_step_up)

        # extract partition function from gauge transforma
        self.partition_function_Z = self.calc_partition_function(self.gauge_transform_g, T)
        # partition function Z is necessary for normalization of reweighted dynamics

        # compute reweighted dynamics from gauge transform
        self.reweighted_dynamics_P_W = self.calc_reweighted_dynamics_array(self.gauge_transform_g, T, s,
                                                                           x_T=x_T, prob_step_up=prob_step_up)


    @property
    def all_init_params_dict(self):
        return {"T": self.T, "s": self.s, "x_T": self.x_T, "prob_step_up": self.prob_step_up}


    @staticmethod
    def calc_weight_function(x: float | np.ndarray, s: float, x_T: int) -> float | np.ndarray:
        """
        Calculate weight function W of (softened) constraint for rare trajectories with a specified end point.

        Parameters:
            x: end point of trajectory
            s: softening parameter
            x_T: required end point of rare trajectory

        Returns:
            weight function value
        """

        assert s >= 0, "s >= 0 required"

        if s == np.inf:
            return np.where(x == x_T, 1, 0)  # delta function

        else:
            return np.exp(- s * (x - x_T) ** 2)


    def calc_gauge_transform_array(self, T: int, s: float, x_T: int, prob_step_up: float) -> np.ndarray:
        """
        Calculate gauge transform g (for rare trajectories of random walk with softened constraint) via
        iterative solution of recursion equation.
        For details see Rose et al. 2021 New J. Phys. 23 013013, https://doi.org/10.1088/1367-2630/abd7bd,
        App. A and in particular Eq. (A.10).

        Parameters:
            T: #time steps of random walk
            s: softening parameter
            x_T: required end point of rare trajectory
            prob_step_up: probability to go 1 step up

        Returns:
            g(x, t) for parameters T and s
        """

        # initialization
        g_array = np.empty((T + 1, 2 * T + 1))
        g_array[:] = np.nan

        # utility function
        def calc_g_value(x: int, t: int):
            if t == T:
                return 1

            elif t == T - 1:
                return prob_step_up * self.calc_weight_function(x + 1, s, x_T=x_T) \
                    + (1 - prob_step_up) * self.calc_weight_function(x - 1, s, x_T=x_T)

            else:
                return prob_step_up * g_array[t + 1, x + 1 + T] \
                    + (1 - prob_step_up) * g_array[t + 1, x - 1 + T]

        # compute gauge transform values
        for t in np.arange(T + 1)[::-1]:
            for x in np.arange(- t, t + 1, 2):
                g_array[t, x + T] = calc_g_value(x, t)

        return g_array


    def calc_partition_function(self, gauge_transform_g: np.ndarray, T: int) -> float:
        """
        Extract partition function Z from gauge transform.
        For details see Rose et al. 2021 New J. Phys. 23 013013, https://doi.org/10.1088/1367-2630/abd7bd,
        App. B and in particular Eq. (B.2).

        Parameters:
            T: #time steps of random walk
            gauge_transform_g: array containing gauge transform values

        Returns:
            partition function Z
        """

        return gauge_transform_g[0, T]  # corresponds to calc_g_value(x = 0, t = 0), see calc_gauge_transform_array


    def calc_reweighted_dynamics_array(self, gauge_transform_g: np.ndarray, T: int, s: float, x_T: int,
                                       prob_step_up: float) -> np.ndarray:
        """
        Calculate reweighted dynamics P_W (for rare trajectories of random walk with softened constraint)
        via a gauge transform.
        For details see Rose et al. 2021 New J. Phys. 23 013013, https://doi.org/10.1088/1367-2630/abd7bd,
        App. A and in particular Eq. (A.9).

        Parameters:
            gauge_transform_g: array containing gauge transform values
            T: #time steps of random walk
            s: softening parameter
            x_T: required end point of rare trajectory
            prob_step_up: probability to go 1 step up

        Returns:
            array containing reweighted dynamics P_W
        """

        # compute weights for 1 step up in last time step
        weights_array = np.ones((T, 2 * T - 1))
        weights_array[T - 1, :] = ReweightedDynamics.calc_weight_function(np.arange(- T + 1, T) + 1, s, x_T=x_T)

        # compute reweighted probabilities to go 1 step up according to Eq. (A.9) in Rose et al. 2021
        P_W_array = gauge_transform_g[1:, 2:] / gauge_transform_g[:-1, 1:-1] * weights_array * prob_step_up
        # P_W_array only contains probabilities to go 1 step up (that to go 1 step down is given by normalization)

        return P_W_array




