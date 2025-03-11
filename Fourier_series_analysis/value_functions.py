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
from reweighted_dynamics import ReweightedDynamics
from utilities import ConsistentParametersClass


class ValueFunction(ConsistentParametersClass):
    """
    Class to compute value function for arbitrary transition probabilities as if used as policy in reinforcement
    learning of rare trajectories of random walk.
    """
    def __init__(self, p_theta_distribution: np.ndarray, T: int, s: float, x_T: int, prob_step_up: float):
        """
        Compute value function for arbitrary transition probabilities as if used as policy in reinforcement learning of
        rare trajectories of random walk.

        Parameters:
            p_theta_distribution: distribution P_theta of arbitrary transition probabilities
                                  (p_theta_distribution[t - 1, position_x + T - 1] is expected to be probability
                                  for transition time_t - 1, position_x --> time_t, position_x + 1 according to P_theta)
            T: #time steps of random walk
            s: reward function parameter (balancing closeness of p_theta_distribution to prob_step_up vs.
                                          probability of generating specific rare trajectories)
            x_T: required end point of rare trajectory
            prob_step_up: probability to go 1 step up

        Returns:
            object of class ValueFunction with value function for arbitrary transition probabilities as if used
            as policy in reinforcement learning
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

        self.p_theta_distribution = p_theta_distribution

        # compute value function array
        self.value_func_array = self.calc_value_function_array(p_theta_distribution, T, s, x_T, prob_step_up)


    @property
    def all_init_params_dict(self):
        return {"T": self.T, "s": self.s, "x_T": self.x_T, "prob_step_up": self.prob_step_up}


    @staticmethod
    def calc_reward(action_a: int, position_x: int, time_t: int, p_theta_distribution: np.ndarray, T: int, s: float,
                    x_T: int, prob_step_up: float) -> float:
        """
        Calculate reward for transition time_t - 1, position_x --> time_t, position_x + action_a according to
        reinforcement learning framework applied to 1D random walk.

        Parameters:
            action_a: action_a of random walker (either -1 or +1, representing left/down or right/up step)
            position_x: position of random walker at time_t - 1
            time_t: time step
            p_theta_distribution: distribution P_theta of arbitrary transition probabilities
                                  (p_theta_distribution[t - 1, position_x + tot_time - 1] is expected to be probability
                                  for transition time_t - 1, position_x --> time_t, position_x + 1 according to P_theta)
            T: #time steps of random walk
            s: reward function parameter (balancing closeness of p_theta_distribution to prob_step_up vs.
                                          probability of generating specific rare trajectories)
            x_T: required end point of rare trajectory
            prob_step_up: probability to go 1 step up

        Returns:
            reward for transition time_t - 1, position_x --> time_t, position_x + action_a
        """

        # asserts
        assert action_a in [-1, 1], "action_a must be either -1 or +1"
        assert time_t >= 0, "time_t >= 0 required"
        assert T > 0, "T > 0 required"
        assert s >= 0., "s >= 0. required"
        assert 0. <= prob_step_up <= 1., "0 <= prob_step_up <= 1 required"

        # calculate weight 
        if time_t == T:
            weight = ReweightedDynamics.calc_weight_function(position_x + action_a, s, x_T)
        else:
            weight = 1

        # probabilities for up/right step
        p_theta = p_theta_distribution[time_t - 1, position_x + T - 1]
        # regarding the indices consider the simplest example T = 2:
        # np.shape(p_theta_distribution) = (2, 3) corresponding to values t = 0, 1, and x = -1, 0, 1

        # calculate reward
        if action_a == 1:
            return np.log(weight) - np.log(p_theta) + np.log(prob_step_up)
        else:
            return np.log(weight) - np.log(1 - p_theta) + np.log(1 - prob_step_up)
        

    def calc_value_function(self, x: int, t: int, p_theta_distribution: np.ndarray, T: int, s: float, x_T: int,
                            prop_step_up: float) -> float:
        """
        Calculate value function for random walk at position_x and time_t via Bellman equation recursively and
        store results in self.value_func_array.
        The value function is defined as the expected sum of rewards for all future transitions starting from
        position_x and time_t up to tot_time.

        Parameters:
            x: position
            t: time
            p_theta_distribution: distribution P_theta of arbitrary transition probabilities
                                  (p_theta_distribution[t - 1, position_x + T - 1] is expected to be probability
                                  for transition time_t - 1, position_x --> time_t, position_x + 1 according to P_theta)
            T: #time steps of random walk
            s: reward function parameter (balancing closeness of p_theta_distribution to prob_step_up vs.
                                          probability of generating specific rare trajectories)
            x_T: required end point of rare trajectory
            prop_step_up: probability to go 1 step up

        Returns:
            value function for random walk at position_x and time_t
        """

        # calculate value function value via Bellman eq.
        if t == T:
            value_func_val = 0.  # boundary condition

        else:
            # probability for up/right step
            p_theta = p_theta_distribution[t, x + T - 1]
            # regarding the indices consider the simplest example T = 2:
            # np.shape(p_theta_distribution) = (2, 3) corresponding to values t = 0, 1, and x = -1, 0, 1

            if p_theta == 0.:
                value_func_val = (1 - p_theta) \
                                 * (self.calc_value_function(x - 1, t + 1, p_theta_distribution, T, s, x_T,
                                                             prop_step_up)
                                    + self.calc_reward(-1, x, t + 1, p_theta_distribution, T, s, x_T,
                                                       prop_step_up))

            elif p_theta == 1.:
                value_func_val = p_theta \
                                 * (self.calc_value_function(x + 1, t + 1, p_theta_distribution, T, s, x_T,
                                                             prop_step_up)
                                    + self.calc_reward(1, x, t + 1, p_theta_distribution, T, s, x_T,
                                                       prop_step_up))

            else:
                value_func_val = p_theta \
                                 * (self.calc_value_function(x + 1, t + 1, p_theta_distribution, T, s, x_T,
                                                             prop_step_up)
                                    + self.calc_reward(1, x, t + 1, p_theta_distribution, T, s, x_T,
                                                       prop_step_up)) \
                                 + (1 - p_theta) \
                                 * (self.calc_value_function(x - 1, t + 1, p_theta_distribution, T, s, x_T,
                                                             prop_step_up)
                                    + self.calc_reward(-1, x, t + 1, p_theta_distribution, T, s, x_T,
                                                       prop_step_up))

        # store value function value
        self.value_func_array[t, x + T - 1] = value_func_val

        return value_func_val


    def calc_value_function_array(self, p_theta_distribution: np.ndarray, T: int, s: float, x_T: int,
                                  prob_step_up: float) -> np.ndarray:
        """
        Calculate value function for random walk at all positions and times via Bellman equation recursively and
        store results in self.value_func_array.

        Parameters:
            p_theta_distribution: distribution P_theta of arbitrary transition probabilities
                                  (p_theta_distribution[t - 1, position_x + T - 1] is expected to be probability
                                  for transition time_t - 1, position_x --> time_t, position_x + 1 according to P_theta)
            T: #time steps of random walk
            s: reward function parameter (balancing closeness of p_theta_distribution to prop_step_up vs.
                                          probability of generating specific rare trajectories)
            x_T: required end point of rare trajectory
            prob_step_up: probability to go 1 step up

        Returns:
            value function for random walk at all positions and times
        """

        # initialization
        self.value_func_array = np.empty((T + 1, 2 * T + 1))
        self.value_func_array[:] = np.nan

        # compute value function
        self.calc_value_function(0, 0, p_theta_distribution, T, s, x_T, prob_step_up)

        return self.value_func_array
