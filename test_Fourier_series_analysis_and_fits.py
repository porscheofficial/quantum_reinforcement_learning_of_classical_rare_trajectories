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
import pytest
from Fourier_series_analysis_and_fits import ParameterizedDynamicsFits


@pytest.mark.parametrize("no_layers, indicator_nonzero_features, original_params_array, expected_amplitudes, expected_phases",
    [
        # 1 layer, 2 nonzero features (including c_00)
        (
            1,
            np.array([[0, 1, 0],
                      [0, 1, 0]]),
            np.arange(6.),  # 2 input scaling parameters, 2 amplitudes, 1 phase, 1 output scaling parameter
            [0.0, 2.0, 0.0,
             0.0, 3.0, 0.0],  # a_array
            [0.0, 0.0, 0.0,
             0.0, 4.0, 0.0],  # phi_array
        ),
        # 1 layer, only c_00 nonzero
        (
            1,
            np.array([[0, 1, 0],
                      [0, 0, 0]]),
            np.arange(4.),  # 2 input scaling parameters, 1 amplitude, 1 output scaling parameter
            [0.0, 2.0, 0.0,
             0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0,
             0.0, 0.0, 0.0],
        ),
        # 2 layers, 3 nonzero features
        (
            2,
            np.array([[0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0]]),
            np.arange(9.),  # 2 input scaling parameters, 3 amplitude, 3 phases, 1 output scaling parameter
            [0.0, 2.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 3.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 4.0, 0.0],
            [0.0, 5.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 6.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 7.0, 0.0],
        ),
    ]
)
def test_calc_params_array_random_Fourier_features_behavior(no_layers, indicator_nonzero_features, original_params_array,
                                                            expected_amplitudes, expected_phases):
    result = ParameterizedDynamicsFits.calc_params_array_random_Fourier_features(no_layers, original_params_array,
                                                                                 indicator_nonzero_features)

    no_pos_freqs = no_layers + 1
    no_freqs = 2 * no_layers + 1
    a_array = result[2:2 + no_pos_freqs * no_freqs]
    phi_array = result[2 + no_pos_freqs * no_freqs:-1]

    np.testing.assert_allclose(a_array, expected_amplitudes)
    np.testing.assert_allclose(phi_array, expected_phases)
