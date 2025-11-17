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


import sys
import numpy as np
import pytest

from pathlib import Path
from timeit import timeit

main_directory = Path(__file__).resolve().parent.parent
sys.path.append(str(main_directory))

from policy_evaluation_and_plots import PolicyEvaluation


def test_use_calc_trajectories_x_array_jit_vs_non_jit():
    """
    Test function for PolicyEvaluation._calc_trajectories_x_array and _use_calc_trajectories_x_array_jit.
    """
    T = 200
    no_trajectories = 20
    no_policies = 2

    rng = np.random.default_rng()
    policies_array = rng.random((no_policies, T, 2 * T - 1))

    num = 1000
    t = timeit(lambda: PolicyEvaluation._calc_trajectories_x_array(policies_array, T, no_trajectories, no_policies),
               number=num)
    t_jit = timeit(lambda: PolicyEvaluation._use_calc_trajectories_x_array_jit(policies_array, T, no_trajectories, no_policies),
                   number=num)

    print(f"Time for executing {num} times:\n"
          f"_calc_trajectories_x_array: {t}\n"
          f"_calc_trajectories_x_array_jit: {t_jit}")


def test_calc_return_values_vectorized_vs_non_vectorized_mode():
    """
    Test function for PolicyEvaluation.calc_return_values in/not in vectorized mode.
    """
    T = 200
    no_trajectories = 20
    no_policies = 2

    s = 1
    x_T = 0
    prob_step_up = 0.5

    rng = np.random.default_rng()
    policies_array = rng.random((no_policies, T, 2 * T - 1))

    trajectories_x_array = PolicyEvaluation._use_calc_trajectories_x_array_jit(policies_array, T, no_trajectories, no_policies)

    result = PolicyEvaluation.calc_return_values(trajectories_x_array, policies_array, s, x_T, prob_step_up,
                                                 use_vectorized_func = False)
    result_vectorized = PolicyEvaluation.calc_return_values(trajectories_x_array, policies_array, s, x_T, prob_step_up,
                                                            use_vectorized_func = True)

    assert np.array_equal(result, result_vectorized)

    num = 1000
    t = timeit(lambda: PolicyEvaluation.calc_return_values(trajectories_x_array, policies_array, s, x_T, prob_step_up,
                                                           use_vectorized_func = False), number=num)
    t_vectorized = timeit(lambda: PolicyEvaluation.calc_return_values(trajectories_x_array, policies_array, s, x_T, prob_step_up,
                                                                      use_vectorized_func = True), number=num)

    print(f"Time for executing {num} times _calc_trajectories_x_array:\n"
          f"not in vectorized mode: {t}\n"
          f"in vectorized mode: {t_vectorized}")
