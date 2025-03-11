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
from multiprocessing import Pool
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import dill
import warnings
from typing import Callable, List, Tuple
from scipy.optimize import minimize
from sympy.physics.quantum import TensorProduct
from logging_config import get_logger
from policy_evaluation_and_plots import PolicyEvaluation
from utilities import einsum_subscripts, ProgressBar, ConsistentParametersClass


logger = get_logger("Fourier_series_analysis_and_fits.py")

# custom warning handler
def log_runtime_warning(message, category, filename, lineno, file=None, line=None):
    logger.warning(f'{category.__name__}: {message} in {filename} at line {lineno}')

# redirect warnings to logger
warnings.showwarning = log_runtime_warning


# constants
sigma_x = np.array([[0., 1.], [1., 0.]])
sigma_y = np.array([[0., - 1j], [1j, 0.]])
sigma_z = np.array([[1., 0.], [0., -1.]])

sp_identity = sp.eye(2)
sp_sigma_x = sp.Matrix([[0, 1], [1, 0]])
sp_sigma_y = sp.Matrix([[0, - 1j], [1j, 0]])
sp_sigma_z = sp.Matrix([[1, 0], [0, -1]])

v_x = 1 / np.sqrt(2) * np.array([[1., 1.], [1., -1.]])
v_x_dagger = v_x


class FourierSeriesAnalysis(ConsistentParametersClass):
    """
    Class for Fourier series analysis of parameterized quantum circuits (PQC).
    """
    def __init__(self, no_qubits: int, no_layers: int, symbolic_or_numeric: str,
                 no_samples_variational_params: int = None, random_thetas: bool = None,
                 deactivate_r_x=False, deactivate_r_y_and_r_z=False, deactivate_ctrl_z=False):
        """
        Calculate Fourier series of expectation value of observable for parameterized quantum circuit (PQC).

        Parameters:
            no_qubits: #qubits of PQC
            no_layers: #data-uploading layers of PQC
            symbolic_or_numeric: "symbolic" or "numeric" (symbolic or numerical calculation of Fourier coefficients)
            no_samples_variational_params: #samples for numerical calculation of Fourier coefficients
            random_thetas: if True, random thetas are used for numerical calculation of Fourier coefficients
            deactivate_r_x: if True, the gates in the PQC for rotations around x-axis are deactivated
            deactivate_r_y_and_r_z: if True, the gates in the PQC for rotations around y- and z-axis are deactivated
            deactivate_ctrl_z: if True, the controlled-Z gate in the PQC is deactivated
                               (only relevant for no_qubits == 2)

        Returns:
            None
        """

        super().__init__()

        # asserts
        assert no_qubits > 0, "no_qubits > 0 required"

        if no_qubits > 2:
            raise NotImplementedError("Case no_qubits > 2 has not been implemented yet.")

        assert no_layers > 0, "no_layers > 0 required"
        assert symbolic_or_numeric in ["symbolic", "numeric"], "symbolic_or_numeric must be 'symbolic' or 'numeric'"

        if symbolic_or_numeric == "numeric":
            assert no_samples_variational_params is not None, "no_samples_variational_params must be provided"
            assert no_samples_variational_params > 0, "no_samples_variational_params > 0 required"
            assert random_thetas is not None, "random_thetas must be provided"

        # save inputs
        self.no_qubits = no_qubits
        self.no_layers = no_layers
        self.symbolic_or_numeric = symbolic_or_numeric

        if symbolic_or_numeric == "numeric":
            self.no_samples_variational_params = no_samples_variational_params
            self.random_thetas = random_thetas

        self.deactivate_r_x = deactivate_r_x
        self.deactive_r_y_and_r_z = deactivate_r_y_and_r_z
        self.deactive_ctrl_z = deactivate_ctrl_z

        # initializations
        if no_qubits == 1:
            obs = sp_sigma_z
        elif no_qubits == 2:
            obs = self.tensor_prod(sp_sigma_z, sp_sigma_z, sympy_expr=True)

        # calculate Fourier coefficients
        if symbolic_or_numeric == "symbolic":
            x, t = sp.symbols("x, t", real=True)
            
            if no_qubits == 1:
                expectation_val_obs = self.calc_expectation_value_1_qubit_n_layers(no_layers, obs,
                                                                                   deactivate_r_x=deactivate_r_x,
                                                                                   deactivate_r_y_and_r_z=deactivate_r_y_and_r_z)
                
            elif no_qubits == 2:
                expectation_val_obs = self.calc_expectation_value_2_qubits_n_layers(no_layers, obs,
                                                                                    deactivate_r_x=deactivate_r_x,
                                                                                    deactivate_r_y_and_r_z=deactivate_r_y_and_r_z,
                                                                                    deactivate_ctrl_z=deactivate_ctrl_z)

            self.coeffs_array, self.exp_series, self.amp_phase_series = \
                self.calc_multivariate_Fourier_series(expectation_val_obs, x, t, no_layers, no_layers)

        elif symbolic_or_numeric == "numeric":
            self.coeffs_samples_array = \
                FourierSeriesAnalysis.calc_Fourier_coeffs_PQC(no_qubits, no_layers, obs,
                                                              no_samples=no_samples_variational_params,
                                                              random_thetas=random_thetas,
                                                              deactivate_r_x=deactivate_r_x,
                                                              deactivate_r_y_and_r_z=deactivate_r_y_and_r_z,
                                                              deactivate_ctrl_z=deactivate_ctrl_z)


    @property
    def all_init_params_dict(self):
        if self.symbolic_or_numeric == "symbolic":
            return {"no_qubits": self.no_qubits, "no_layers": self.no_layers}

        elif self.symbolic_or_numeric == "numeric":
            return {"no_qubits": self.no_qubits, "no_layers": self.no_layers,
                    "no_samples_variational_params": self.no_samples_variational_params}


    @staticmethod
    def param_subs(expr: sp.Expr | sp.Matrix, subs_dict: dict, symbolic: bool) \
            -> sp.Expr | sp.Matrix | float | np.ndarray:
        """
        Substitute parameters in an expression or matrix.

        Parameters:
            expr: expression or matrix
            subs_dict: dictionary with substitutions
            symbolic: if True, the resulting expression is symbolic (SymPy), else numerical

        Returns:
            sp.Expr or sp.Matrix or float or np.ndarray
        """

        if symbolic:
            return expr.subs(subs_dict)
        else:
            return expr.evalf(subs=subs_dict, chop=1e-16)


    @staticmethod
    def tensor_prod(a: sp.Expr | np.ndarray, b: sp.Expr | np.ndarray, sympy_expr=False) -> sp.Expr | np.ndarray:
        """
        Calculate tensor product of two matrices,
        if sympy_expr == True the symbolic (SymPy) expressions, else the numerical ones.

        Parameters:
            a: first matrix
            b: second matrix
            sympy_expr: see description above

        Returns:
            sp.Expr or np.ndarray
        """

        if sympy_expr:
            return TensorProduct(a, b)
        else:
            return np.kron(a, b)


    @staticmethod
    def r_n(alpha: float | sp.Symbol, n: str, sympy_expr=False, deactivated=False) -> sp.Expr | np.ndarray:
        """
        Calculate rotations on Bloch sphere,
        if sympy_expr == True the symbolic (SymPy) expressions, else the numerical ones.

        Parameters:
            alpha: rotation angle
            n: rotation axis, either "x" or "y" or "z"
            sympy_expr: see description above
            deactivated: if True, the gate is deactivated (i.e., the identity matrix is returned)

        Returns:
            sp.Expr or np.ndarray
        """

        identity = np.identity(2) if not sympy_expr else sp_identity

        if deactivated:
            return identity

        if n == "x":
            sigma = sigma_x if not sympy_expr else sp_sigma_x
        elif n == "y":
            sigma = sigma_y if not sympy_expr else sp_sigma_y
        elif n == "z":
            sigma = sigma_z if not sympy_expr else sp_sigma_z
        else:
            raise ValueError("n must be 'x' or 'y' or 'z'")

        if sympy_expr:
            return sp.cos(alpha / 2) * identity - 1j * sp.sin(alpha / 2) * sigma
        else:
            return np.cos(alpha / 2) * identity - 1j * np.sin(alpha / 2) * sigma


    @staticmethod
    def r_n_multi_qubit(tot_no_qubits: int, acting_on_qubit_no: int, alpha: float | sp.Symbol, n: str, sympy_expr=False,
                        deactivated=False) -> sp.Expr | np.ndarray:
        """
        Calculate tensor product of rotations on Bloch sphere for multiple qubits,
        if sympy_expr == True the symbolic (SymPy) expressions, else the numerical ones.

        Parameters:
            tot_no_qubits: total number of qubits
            acting_on_qubit_no: number of qubit on which the rotation acts
            alpha: rotation angle
            n: rotation axis, either "x" or "y" or "z"
            sympy_expr: see description above
            deactivated: if True, the gate is deactivated (i.e., the identity matrix is returned)

        Returns:
            sp.Expr or np.ndarray
        """

        assert tot_no_qubits > 1, "tot_no_qubits must be greater than 1"

        identity = np.identity(2) if not sympy_expr else sp_identity
        single_qubit_r_n = FourierSeriesAnalysis.r_n(alpha, n, sympy_expr=sympy_expr, deactivated=deactivated)

        def choose_nth_matrix(n):
            return single_qubit_r_n if acting_on_qubit_no == n else identity

        matrix = FourierSeriesAnalysis.tensor_prod(choose_nth_matrix(1), choose_nth_matrix(2), sympy_expr=sympy_expr)

        for n in range(3, tot_no_qubits + 1):
            matrix = FourierSeriesAnalysis.tensor_prod(matrix, choose_nth_matrix(n), sympy_expr=sympy_expr)

        return matrix


    @staticmethod
    def ctrl_z(sympy_expr=False, deactivated=False) -> sp.Expr | np.ndarray:
        """
        Calculate controlled-Z gate for 2 qubits,
        if sympy_expr == True the symbolic (SymPy) expressions, else the numerical ones.

        Parameters:
            sympy_expr: see description above
            deactivated: if True, the gate is deactivated (i.e., the identity matrix is returned)

        Returns:
            sp.Expr or np.ndarray
        """
        if sympy_expr:
            cz = sp.eye(4)
        else:
            cz = np.identity(4)

        if deactivated:
            return cz

        cz[3, 3] = -1.
        return cz


    @staticmethod
    def calc_unitary_transform_1_qubit_1_layer(thetas: np.ndarray, four_thetas: bool, dagger: bool,
                                               deactivate_r_x=False, deactivate_r_y_and_r_z=False) -> sp.Matrix:
        """
        Calculate unitary transform symbolically for parameterized quantum circuit with 1 qubit and 1 data-uploading
        layer.

        Parameters:
            thetas: angles of rotations/variational parameters
            four_thetas: if True, thetas has length 4, else length 3
            dagger: if True, the unitary transform is the adjoint of the original one
            deactivate_r_x: if True, the gates in the PQC for rotations around x-axis are deactivated
            deactivate_r_y_and_r_z: if True, the gates in the PQC for rotations around y- and z-axis are deactivated

        Returns:
            unitary transform
        """

        # asserts
        if four_thetas:
            assert len(thetas) == 4, "if four_thetas == True, thetas must be of length 4"
        else:
            assert len(thetas) == 3, "if four_thetas == False, thetas must be of length 3"

        # calculate unitary transform for 1 layer
        x, t = sp.symbols("x, t", real=True)

        # utility functions
        def r_x(alpha):
            return FourierSeriesAnalysis.r_n(alpha, "x", sympy_expr=True, deactivated=deactivate_r_x)

        def r_y(alpha):
            return FourierSeriesAnalysis.r_n(alpha, "y", sympy_expr=True, deactivated=deactivate_r_y_and_r_z)

        def r_z(alpha):
            return FourierSeriesAnalysis.r_n(alpha, "z", sympy_expr=True, deactivated=deactivate_r_y_and_r_z)


        if not dagger:
            unitary_transform = (r_y(thetas[2]) @ r_x(x)
                                 @ r_z(thetas[1]) @ r_y(thetas[0]) @ r_x(t))

            if four_thetas:
                unitary_transform = r_z(thetas[3]) @ unitary_transform

        else:
            unitary_transform = (r_x(-t) @ r_y(-thetas[0]) @ r_z(-thetas[1])
                                 @ r_x(-x) @ r_y(-thetas[2]))

            if four_thetas:
                unitary_transform = unitary_transform @ r_z(-thetas[3])

        return unitary_transform


    @staticmethod
    def calc_unitary_transform_2_qubits_1_layer(thetas: np.ndarray, four_thetas: bool, dagger: bool,
                                                deactivate_r_x=False, deactivate_r_y_and_r_z=False,
                                                deactivate_ctrl_z=False) -> sp.Matrix:
        """
        Calculate unitary transform symbolically for parameterized quantum circuit with 2 qubits and 1 data-uploading
        layer.

        Parameters:
            thetas: angles of rotations/variational parameters
            four_thetas: if True, thetas has length 4, else length 3
            dagger: if True, the unitary transform is the adjoint of the original one
            deactivate_r_x: if True, the gates in the PQC for rotations around x-axis are deactivated
            deactivate_r_y_and_r_z: if True, the gates in the PQC for rotations around y- and z-axis are deactivated
            deactivate_ctrl_z: if True, the controlled-Z gate in the PQC is deactivated

        Returns:
            unitary transform
        """

        # asserts
        if four_thetas:
            assert len(thetas) == 4, "if four_thetas == True, thetas must be of length 4"
        else:
            assert len(thetas) == 2, "if four_thetas == False, thetas must be of length 3"

        # compute unitary transform for 1 layer
        x, t = sp.symbols("x, t", real=True)

        # utility functions
        def r_x_2_qubits(acting_on_qubit_no, alpha):
            return FourierSeriesAnalysis.r_n_multi_qubit(2, acting_on_qubit_no, alpha, "x", sympy_expr=True,
                                                         deactivated=deactivate_r_x)

        def r_y_2_qubits(acting_on_qubit_no, alpha):
            return FourierSeriesAnalysis.r_n_multi_qubit(2, acting_on_qubit_no, alpha, "y", sympy_expr=True,
                                                         deactivated=deactivate_r_y_and_r_z)

        def r_z_2_qubits(acting_on_qubit_no, alpha):
            return FourierSeriesAnalysis.r_n_multi_qubit(2, acting_on_qubit_no, alpha, "z", sympy_expr=True,
                                                         deactivated=deactivate_r_y_and_r_z)

        ctrl_z = FourierSeriesAnalysis.ctrl_z(sympy_expr=True, deactivated=deactivate_ctrl_z)

        if not dagger:
            unitary_transform = (r_y_2_qubits(2, thetas[1])
                                 @ r_y_2_qubits(1, thetas[0])
                                 @ r_x_2_qubits(2, x)
                                 @ r_x_2_qubits(1, t))

            if four_thetas:
                unitary_transform = (ctrl_z
                                     @ r_z_2_qubits(2, thetas[3])
                                     @ r_z_2_qubits(1, thetas[2])
                                     @ unitary_transform)

        else:
            unitary_transform = (r_x_2_qubits(1, -t)
                                 @ r_x_2_qubits(2, -x)
                                 @ r_y_2_qubits(1, -thetas[0])
                                 @ r_y_2_qubits(2, -thetas[1]))

            if four_thetas:
                unitary_transform = (unitary_transform @
                                     r_z_2_qubits(1, -thetas[2])
                                     @ r_z_2_qubits(2, -thetas[3])
                                     @ ctrl_z)

        return unitary_transform


    @staticmethod
    def calc_expectation_value_1_qubit_n_layers(no_layers: int, obs: sp.Matrix | np.ndarray,
                                                theta_vals: np.ndarray = None, deactivate_r_x=False,
                                                deactivate_r_y_and_r_z=False) \
            -> sp.Matrix | np.ndarray:
        """
        Compute expectation value of observable for parameterized quantum circuit with 1 qubit and n data-uploading
        layers.

        Parameters:
            no_layers: number of data-uploading layers
            obs: observable
            theta_vals: angles of rotations/variational parameters; if None, thetas are symbolic, otherwise numerical
            deactivate_r_x: if True, the gates in the PQC for rotations around x-axis are deactivated
            deactivate_r_y_and_r_z: if True, the gates in the PQC for rotations around y- and z-axis are deactivated

        Returns:
            expectation value
        """

        # asserts
        assert no_layers > 0, "no_layers > 0 is required"
        assert np.shape(obs) == (2, 2) and np.all(np.conj(np.transpose(obs)) == np.array(obs)),\
            "obs must be a Hermitian 2x2-matrix"

        if theta_vals is not None:
            assert len(theta_vals) == 4 * no_layers - 1, "thetas must be of length 4 * no_layers - 1"

        # compute generic forms of unitary transforms for layers
        alpha, beta, gamma, delta = sp.symbols("alpha, beta, gamma, delta", real=True)
        generic_thetas = np.array([alpha, beta, gamma, delta])

        if np.all(np.array(obs, dtype=float) == sigma_z):
            unitary_last_layer = \
                FourierSeriesAnalysis.calc_unitary_transform_1_qubit_1_layer(generic_thetas[:3], False, False,
                                                                             deactivate_r_x=deactivate_r_x,
                                                                             deactivate_r_y_and_r_z=deactivate_r_y_and_r_z)
            unitary_last_layer_dagger = \
                FourierSeriesAnalysis.calc_unitary_transform_1_qubit_1_layer(generic_thetas[:3], False, True,
                                                                             deactivate_r_x=deactivate_r_x,
                                                                             deactivate_r_y_and_r_z=deactivate_r_y_and_r_z)

        if not np.all(np.array(obs, dtype=float) == sigma_z) or no_layers > 1:
            unitary_1_layer = \
                FourierSeriesAnalysis.calc_unitary_transform_1_qubit_1_layer(generic_thetas, True, False,
                                                                             deactivate_r_x=deactivate_r_x,
                                                                             deactivate_r_y_and_r_z=deactivate_r_y_and_r_z)
            unitary_1_layer_dagger = \
                FourierSeriesAnalysis.calc_unitary_transform_1_qubit_1_layer(generic_thetas, True, True,
                                                                             deactivate_r_x=deactivate_r_x,
                                                                             deactivate_r_y_and_r_z=deactivate_r_y_and_r_z)

        # multiply unitary transforms for layers
        if theta_vals is None:
            no_thetas = 4 * no_layers - 1
            thetas = sp.symbols("theta1:" + str(no_thetas + 1), real=True)
            symbolic = True
        else:
            thetas = theta_vals
            symbolic = False

        unitary = np.identity(2)
        unitary_dagger = np.identity(2)

        if np.all(np.array(obs, dtype=float) == sigma_z):
            n_max = no_layers - 1
        else:
            n_max = no_layers


        for n in range(n_max):
            subs_dict = {alpha: thetas[4 * n + 0], beta: thetas[4 * n + 1],
                         gamma: thetas[4 * n + 2], delta: thetas[4 * n + 3]}

            unitary = FourierSeriesAnalysis.param_subs(unitary_1_layer, subs_dict, symbolic) @ unitary
            unitary_dagger = unitary_dagger @ FourierSeriesAnalysis.param_subs(unitary_1_layer_dagger, subs_dict,
                                                                               symbolic)

        if np.all(np.array(obs, dtype=float) == sigma_z):
            subs_dict = {alpha: thetas[-3], beta: thetas[-2], gamma: thetas[-1]}

            unitary = FourierSeriesAnalysis.param_subs(unitary_last_layer, subs_dict, symbolic) @ unitary
            unitary_dagger = unitary_dagger @ FourierSeriesAnalysis.param_subs(unitary_last_layer_dagger, subs_dict,
                                                                               symbolic)

        expectation_val = (unitary_dagger @ obs @ unitary)[0, 0]
        # [0, 0] because initial state of quantum circuit is |0>

        return expectation_val


    @staticmethod
    def calc_expectation_value_2_qubits_n_layers(no_layers: int, obs: sp.Matrix | np.ndarray,
                                                 theta_vals: np.ndarray = None, deactivate_r_x=False,
                                                 deactivate_r_y_and_r_z=False, deactivate_ctrl_z=False) \
            -> sp.Matrix | np.ndarray:
        """
        Compute expectation value of observable for parameterized quantum circuit with 1 qubit and n data-uploading
        layers.

        Parameters:
            no_layers: number of data-uploading layers
            obs: observable
            theta_vals: angles of rotations/variational parameters; if None, thetas are symbolic, otherwise numerical
            deactivate_r_x: if True, the gates in the PQC for rotations around x-axis are deactivated
            deactivate_r_y_and_r_z: if True, the gates in the PQC for rotations around y- and z-axis are deactivated
            deactivate_ctrl_z: if True, the controlled-Z gate in the PQC is deactivated

        Returns:
            expectation value
        """

        # asserts
        assert no_layers > 0, "no_layers > 0 is required"
        assert np.shape(obs) == (4, 4) and np.all(np.conj(np.transpose(obs)) == np.array(obs)), \
            "obs must be a Hermitian 4x4-matrix"

        if theta_vals is not None:
            assert len(theta_vals) == 4 * no_layers - 2, "thetas must be of length 4 * no_layers - 2"

        # compute generic forms of unitary transforms for layers
        alpha, beta, gamma, delta = sp.symbols("alpha, beta, gamma, delta", real=True)
        generic_thetas = np.array([alpha, beta, gamma, delta])

        if np.all(np.array(obs, dtype=float) == FourierSeriesAnalysis.tensor_prod(sigma_z, sigma_z)):
            unitary_last_layer = \
                FourierSeriesAnalysis.calc_unitary_transform_2_qubits_1_layer(generic_thetas[:2], False, False,
                                                                              deactivate_r_x=deactivate_r_x,
                                                                              deactivate_r_y_and_r_z=deactivate_r_y_and_r_z,
                                                                              deactivate_ctrl_z=deactivate_ctrl_z)
            unitary_last_layer_dagger = \
                FourierSeriesAnalysis.calc_unitary_transform_2_qubits_1_layer(generic_thetas[:2], False, True,
                                                                              deactivate_r_x=deactivate_r_x,
                                                                              deactivate_r_y_and_r_z=deactivate_r_y_and_r_z,
                                                                              deactivate_ctrl_z=deactivate_ctrl_z)

        if not np.all(np.array(obs, dtype=float) == FourierSeriesAnalysis.tensor_prod(sigma_z, sigma_z)) \
                or no_layers > 1:
            unitary_1_layer = \
                FourierSeriesAnalysis.calc_unitary_transform_2_qubits_1_layer(generic_thetas, True, False,
                                                                              deactivate_r_x=deactivate_r_x,
                                                                              deactivate_r_y_and_r_z=deactivate_r_y_and_r_z,
                                                                              deactivate_ctrl_z=deactivate_ctrl_z)
            unitary_1_layer_dagger = \
                FourierSeriesAnalysis.calc_unitary_transform_2_qubits_1_layer(generic_thetas, True, True,
                                                                              deactivate_r_x=deactivate_r_x,
                                                                              deactivate_r_y_and_r_z=deactivate_r_y_and_r_z,
                                                                              deactivate_ctrl_z=deactivate_ctrl_z)

        # multiply unitary transforms for layers
        if theta_vals is None:
            no_thetas = 4 * no_layers - 2
            thetas = sp.symbols("theta1:" + str(no_thetas + 1), real=True)
            symbolic = True
        else:
            thetas = theta_vals
            symbolic = False

        unitary = np.identity(4)
        unitary_dagger = np.identity(4)

        if np.all(np.array(obs, dtype=float) == FourierSeriesAnalysis.tensor_prod(sigma_z, sigma_z)):
            n_max = no_layers - 1
        else:
            n_max = no_layers


        for n in range(n_max):
            subs_dict = {alpha: thetas[4 * n + 0], beta: thetas[4 * n + 1],
                         gamma: thetas[4 * n + 2], delta: thetas[4 * n + 3]}

            unitary = FourierSeriesAnalysis.param_subs(unitary_1_layer, subs_dict, symbolic) @ unitary
            unitary_dagger = unitary_dagger @ FourierSeriesAnalysis.param_subs(unitary_1_layer_dagger, subs_dict,
                                                                               symbolic)

        if np.all(np.array(obs, dtype=float) == FourierSeriesAnalysis.tensor_prod(sigma_z, sigma_z)):
            subs_dict = {alpha: thetas[-2], beta: thetas[-1]}

            unitary = FourierSeriesAnalysis.param_subs(unitary_last_layer, subs_dict, symbolic) @ unitary
            unitary_dagger = unitary_dagger @ FourierSeriesAnalysis.param_subs(unitary_last_layer_dagger, subs_dict,
                                                                               symbolic)

        expectation_val = (unitary_dagger @ obs @ unitary)[0, 0]
        # [0, 0] because initial state of quantum circuit is |0>

        return expectation_val


    @staticmethod
    def calc_multivariate_Fourier_series(f: sp.Expr, x: sp.Symbol, y: sp.Symbol, m_max: int, n_max: int) \
            -> tuple[sp.Expr, np.ndarray]:
        """
        Calculate exponential and amplitude-phase form of Fourier series of real-valued 2*pi-periodic function
        of two variables x and y.

        Parameters:
            f: function whose Fourier series is to be calculated
            x: first variable with respect to which to calculate the Fourier series
            y: second variable ------------------------"-------------------------
            m_max: max. frequency of first variable in the Fourier series
            n_max: max. frequency of second variable ---------"----------

        Returns:
            calculated complex-valued Fourier coefficients,
            exponential form of the Fourier series of the function,
            amplitude-phase form of the Fourier series of the function
        """

        # asserts
        assert m_max >= 0, "m_max >= 0 required"
        assert n_max >= 0, "n_max >= 0 required"

        # initialization
        coeffs = np.zeros((m_max + 1, 2 * n_max + 1), dtype=object)
        exp_series = 0
        amp_phase_series = 0

        # calculate complex-valued Fourier coefficients c_mn and save them in array coeffs
        for m in range(m_max + 1):
            # only positive frequencies m required due to symmetry of Fourier coefficients when function is real-valued

            for n in range(-n_max, n_max + 1):
                if m == 0 and n < 0:
                    continue

                c_mn = 1 / (2 * sp.pi) ** 2 \
                       * sp.integrate(sp.integrate(f * sp.exp(- sp.I * (m * x + n * y)),
                                                   (x, -sp.pi, sp.pi)),
                                      (y, -sp.pi, sp.pi))

                """
                # correct floating-point errors
                try:
                    if sp.re(c_mn) < 1e-16:
                        c_mn = 1j * sp.im(c_mn)
                except TypeError:
                    pass

                try:
                    if sp.im(c_mn) < 1e-16:
                        c_mn = sp.re(c_mn)
                except TypeError:
                    pass
                """
                c_mn = sp.simplify(c_mn)

                coeffs[m, n + n_max] = c_mn
                # shift second index as positive frequencies are saved in second half of array

                exp_series += c_mn * sp.exp(1j * (m * x + n * y))
                exp_series += sp.conjugate(c_mn) * sp.exp(- 1j * (m * x + n * y))

                if m == 0 and n > 0:
                    coeffs[m, n] = sp.conjugate(c_mn)
                    # complex conjugate for negative frequencies to ensure that function is real-valued

                # convert complex-valued Fourier coefficient to amplitude a_mn and phase phi_mn
                if m == 0 and n == 0:
                    a_mn = c_mn
                    phi_mn = 0.  # phase can be set to 0 for constant term in Fourier series
                else:
                    a_mn = 2 * sp.Abs(c_mn)
                    phi_mn = sp.arg(c_mn)
                    phi_mn = 0. if phi_mn == sp.nan else phi_mn

                amp_phase_series += a_mn * sp.cos(m * x + n * y + phi_mn)

        return coeffs, exp_series, amp_phase_series


    @staticmethod
    def calc_Fourier_coeffs(func: sp.Expr, x: sp.Symbol, y: sp.Symbol, n: int) -> np.ndarray:
        """
        Computes first (2*n+1)^2 Fourier coefficients of 2*pi periodic function func(x, y) in x and y by means of a 2D
        discrete Fourier Transform (which in turn is calculated by means of a 2D Fast Fourier Transform).
        This function consists of code adjusted from the PennyLane demo by Schuld and Meyer
        (https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series/#part-iii-sampling-fourier-coefficients).

        Parameters:
            func: function whose Fourier coefficients are to be calculated
            x: first variable
            y: second variable
            n: parameter determining #Fourier coefficients to be calculated

        Returns:
            Fourier coefficients
        """

        lambdified_func = sp.lambdify([x, y], func, "numpy")

        n_coeffs = 2 * n + 1

        vals = np.linspace(0, 2 * np.pi, n_coeffs, endpoint=False)
        x_mesh, y_mesh = np.meshgrid(vals, vals, indexing="ij")

        z = np.fft.rfftn(np.array(lambdified_func(x_mesh, y_mesh), dtype=np.float64)) / vals.size ** 2

        return np.fft.fftshift(z, axes=0)  # such that the zero-frequency coefficient is in the middle of the array


    @staticmethod
    def calc_Fourier_coeffs_PQC(no_qubits: int, no_layers: int, obs: np.ndarray, no_samples=100,
                                random_thetas=True, deactivate_r_x=False, deactivate_r_y_and_r_z=False,
                                deactivate_ctrl_z=False) -> np.ndarray:
        """
        Calculate Fourier coefficients of Fourier series of expectation value of an observable for specific
        parameterized quantum circuits (PQC) considered here.
        This function consists of code adjusted from the PennyLane demo by Schuld and Meyer
        (https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series/#part-iii-sampling-fourier-coefficients).

        Parameters:
            no_qubits: #qubits of PQC
            no_layers: #data-uploading layers of PQC
            obs: observable in PQC
            no_samples: #sets of values for parameters theta of PQC
                        (might differ from actually used #sets if random_thetas == True)
            random_thetas: if True: parameters theta are chosen randomly; else: regularly spaced
            deactivate_r_x: if True, the gates in the PQC for rotations around x-axis are deactivated
            deactivate_r_y_and_r_z: if True, the gates in the PQC for rotations around y- and z-axis are deactivated
            deactivate_ctrl_z: if True, the controlled-Z gate in the PQC is deactivated
                               (only relevant for no_qubits == 2)

        Returns:
            Fourier coefficients
        """

        # asserts
        assert no_qubits > 0, "no_qubits > 0 required"

        if no_qubits > 2:
            raise NotImplementedError("Case no_qubits > 2 has not been implemented yet.")

        assert no_layers > 0, "no_layers > 0 required"

        assert no_samples > 0, "no_samples > 0 required"

        # initializations
        x, t = sp.symbols("x, t", real=True)
        coeffs_samples = []

        # choose parameters theta
        if no_qubits == 1:
            no_thetas = 4 * no_layers - 1

        elif no_qubits == 2:
            no_thetas = 4 * no_layers - 2

        if random_thetas:
            thetas_array = 2 * np.pi * np.random.rand(no_samples, no_thetas)

        else:
            theta_vals_array = np.linspace(0., 2 * np.pi, num=int(no_samples ** (1 / no_thetas)), endpoint=False)

            theta_vals_array_list = [theta_vals_array] * no_thetas
            theta_meshgrids_list = np.meshgrid(*theta_vals_array_list, indexing="ij")

            thetas_array = np.vstack([meshgrid.ravel() for meshgrid in theta_meshgrids_list]).T

        # calculate expectation value of observable and Fourier coefficients
        no_samples = len(thetas_array)

        progress_bar = ProgressBar(no_samples, f"Fourier coefficients of PQC with #qubits: {no_qubits}, "
                                               f"#layers: {no_layers}, and for #samples: {no_samples}")

        for i in range(no_samples):
            progress_bar.update(i)

            if no_qubits == 1:
                expectation_val_obs = \
                    FourierSeriesAnalysis.calc_expectation_value_1_qubit_n_layers(no_layers, obs,
                                                                                  theta_vals=thetas_array[i],
                                                                                  deactivate_r_x=deactivate_r_x,
                                                                                  deactivate_r_y_and_r_z=deactivate_r_y_and_r_z)

            elif no_qubits == 2:
                expectation_val_obs = \
                    FourierSeriesAnalysis.calc_expectation_value_2_qubits_n_layers(no_layers, obs,
                                                                                   theta_vals=thetas_array[i],
                                                                                   deactivate_r_x=deactivate_r_x,
                                                                                   deactivate_r_y_and_r_z=deactivate_r_y_and_r_z,
                                                                                   deactivate_ctrl_z=deactivate_ctrl_z)

            coeffs_sample = FourierSeriesAnalysis.calc_Fourier_coeffs(expectation_val_obs, x, t, no_layers)
            coeffs_samples.append(coeffs_sample)

        progress_bar.finish()

        return np.array(coeffs_samples)


class ParameterizedDynamicsFits(ConsistentParametersClass):
    """
    Class for fitting the parameterized dynamics (policy of a quantum reinforcement learning algorithm) to
    an optimal dynamics.
    """
    def __init__(self, optimal_dynamics: np.ndarray, no_qubits: int, no_layers: int, no_fits: int,
                 fitting_parameters: str, cost_func_type: str, no_trajectories_cost_func: int = None,
                 max_optimization_steps: int = None, T: int = None, s: float = None, x_T: float = None,
                 prob_step_up: float = None, optimal_average_return: float = None, optimized_params: np.ndarray = None,
                 optimized_no_layers: int = None, compute_in_parallel=False):
        """
        Fit parameterized dynamics to optimal dynamics.

        Parameters:
            optimal_dynamics: optimal dynamics to which the parameterized dynamics is to be fitted
            no_qubits: #qubits of parameterized quantum circuit (PQC)
            no_layers: #layers of PQC
            no_fits: #fits to be performed (each with independent and identically distributed initial guesses for
                                            the fitting parameters)
            fitting_parameters: fitting parameters to be used
                                (currently fitting_parameters == "Fourier_coefficients" and "variational_angles"
                                 implemented)
            cost_func_type: type of cost function to be minimized
                            (currently cost_func_type == "leastsq" and "trajectory_KL_divergence" implemented)
            no_trajectories_cost_func: #trajectories for cost_func_type == "trajectory_KL_divergence"
            max_optimization_steps: maximum number of optimization steps for each fit
            T: #time steps of random walk/trajectories for cost_func_type == "trajectory_KL_divergence"
            s: parameter in reward of trajectories for cost_func_type == "trajectory_KL_divergence"
            x_T: required end point of trajectories for cost_func_type == "trajectory_KL_divergence"
            prob_step_up: probability of 1 step up in original random walk, required to compute rewards for
                          cost_func_type == "trajectory_KL_divergence"
            optimal_average_return: average return of optimal dynamics for cost_func_type == "trajectory_KL_divergence"
            optimized_params: initial guess for fitting parameters already optimized in the case of PQC
                                             with optimized_no_layers-many data-uploading layers
            optimized_no_layers: see optimized_params_fourier_coeffs
            compute_in_parallel: if True, the fits are computed in parallel

        Returns:
            None
        """

        super().__init__()

        # asserts
        assert T > 0, "T > 0 required"
        assert s > 0., "s > 0 required"
        assert no_qubits > 0, "no_qubits > 0 required"

        if no_qubits > 2:
            raise NotImplementedError("Case no_qubits > 2 has not been implemented yet.")

        assert no_layers > 0, "no_layers > 0 required"

        assert no_fits > 0, "no_fits > 0 required"

        if not fitting_parameters in ["Fourier_coefficients", "variational_angles"]:
            raise NotImplementedError('Other options than fitting_parameters == "Fourier_coefficients" or '
                                      '"variational_angles" have not been implemented yet.')

        if max_optimization_steps is not None:
            assert max_optimization_steps > 0, "max_optimization_steps > 0 required"

        if not cost_func_type in ["leastsq", "trajectory_KL_divergence"]:
            raise NotImplementedError('Other options than cost_func_type == "leastsq" or "trajectory_KL_divergence" '
                                      'have not been implemented yet.')

        if cost_func_type == "trajectory_KL_divergence":
            assert no_trajectories_cost_func is not None, \
                'if cost_func_type == "trajectory_KL_divergence", no_trajectories_cost_func must be provided'
            assert no_trajectories_cost_func > 0, \
                'no_trajectories_cost_func > 0 required'
            assert T is not None, 'T must be provided'
            assert s is not None, 's must be provided'
            assert x_T is not None, 'x_T must be provided'
            assert prob_step_up is not None, 'prob_step_up must be provided'
            assert optimal_average_return is not None, 'optimal_average_return must be provided'


        # save inputs
        self.no_qubits = no_qubits
        self.no_layers = no_layers
        self.no_fits = no_fits
        self.cost_func_type = cost_func_type

        if cost_func_type == "trajectory_KL_divergence":
            self.no_trajectories_cost_func = no_trajectories_cost_func
            self.T = T
            self.s = s
            self.x_T = x_T
            self.prob_step_up = prob_step_up

        self.optimal_dynamics = optimal_dynamics
        self.fitting_parameters = fitting_parameters
        self.optimized_params = optimized_params
        self.optimized_no_layers = optimized_no_layers


        # initializations
        t_values = np.arange(T)
        x_values = np.arange(- T + 1, T)

        self.coords_array = np.array([[(t, x) for x in x_values]
                                      for t in t_values])


        # fits in different cases (for):
        constant_args = (self.coords_array, self.optimal_dynamics,
                   self.no_qubits, self.no_layers, self.no_fits, self.cost_func_type)
        constant_kwargs = {"no_trajectories_cost_func": no_trajectories_cost_func, "T": T, "s": s, "x_T": x_T,
                           "prob_step_up": prob_step_up, "optimal_average_return": optimal_average_return,
                           "optimized_params": optimized_params, "optimized_no_layers": optimized_no_layers,
                           "compute_in_parallel": compute_in_parallel, "max_optimization_steps": max_optimization_steps}


        # fits in terms of variational parameters thetas
        if fitting_parameters == "variational_angles":
            # calculate softmax policy as function of variational parameters thetas
            if no_qubits == 1:
                obs = sp_sigma_z

                self.softmax_policy_lambdified = \
                    self.softmax_policy_function_from_sympy_expr(
                        FourierSeriesAnalysis.calc_expectation_value_1_qubit_n_layers(self.no_layers, obs),
                        self.no_layers)

            elif no_qubits == 2:
                obs = FourierSeriesAnalysis.tensor_prod(sp_sigma_z, sp_sigma_z, sympy_expr=True)

                self.softmax_policy_lambdified = \
                    self.softmax_policy_function_from_sympy_expr(
                        FourierSeriesAnalysis.calc_expectation_value_2_qubits_n_layers(self.no_layers, obs),
                        self.no_layers)

            self.fitted_policies_array, self.optimized_params_min = \
                self.fit_policy(self.softmax_policy_from_lambdified_expr,
                                *constant_args, **constant_kwargs,
                                no_thetas=(4 * self.no_layers - 1))


        ## fits in terms of Fourier coefficients (amplitudes and phases)
        elif fitting_parameters == "Fourier_coefficients":
            no_pos_freqs = no_layers + 1
            no_freqs = 2 * no_layers + 1

            if no_layers == 1:
                if no_qubits == 1:
                    softmax_policy_1_qubit_1_layer_Fourier_coeffs = lambda coords_array, params_1_qubit_array: \
                        self.softmax_policy_Fourier_coeffs(
                            coords_array, self.calc_params_array_n_qubits_1_layer(1, params_1_qubit_array,
                                                                                  in_terms_of_thetas=False)
                        )

                    self.fitted_policies_array, self.optimized_params_min = \
                        self.fit_policy(softmax_policy_1_qubit_1_layer_Fourier_coeffs,
                                        *constant_args, **constant_kwargs,
                                        no_amplitudes=3, no_phases=3)

                elif no_qubits == 2:
                    softmax_policy_2_qubits_1_layer_Fourier_coeffs = lambda coords_array, params_2_qubits_array: \
                        self.softmax_policy_Fourier_coeffs(
                            coords_array, self.calc_params_array_n_qubits_1_layer(2, params_2_qubits_array,
                                                                                  in_terms_of_thetas=False)
                        )

                    self.fitted_policies_array, self.optimized_params_min = \
                        self.fit_policy(softmax_policy_2_qubits_1_layer_Fourier_coeffs,
                                        *constant_args, **constant_kwargs,
                                        no_amplitudes=1, no_phases=0)

            else:
                self.fitted_policies_array, self.optimized_params_min = \
                    self.fit_policy(self.softmax_policy_Fourier_coeffs,
                                    *constant_args, **constant_kwargs,
                                    no_amplitudes=no_pos_freqs * no_freqs, no_phases=no_pos_freqs * no_freqs)


    @property
    def all_init_params_dict(self):
        if self.cost_func_type == "leastsq":
            return {"no_qubits": self.no_qubits, "no_layers": self.no_layers, "no_fits": self.no_fits,
                    "fitting_parameters": self.fitting_parameters, "cost_func_type": self.cost_func_type}

        elif self.cost_func_type == "trajectory_KL_divergence":
            return {"no_qubits": self.no_qubits, "no_layers": self.no_layers, "no_fits": self.no_fits,
                    "fitting_parameters": self.fitting_parameters, "cost_func_type": self.cost_func_type,
                    "no_trajectories_cost_func": self.no_trajectories_cost_func, "T": self.T, "s": self.s,
                    "x_T": self.x_T, "prob_step_up": self.prob_step_up}


    @staticmethod
    def get_subarrays_from_params_array(params_array: np.ndarray, no_layers: int) -> tuple[np.ndarray]:
        """
        Get subarrays of variational parameters (lambda_array, amplitudes_array, phases_array, and w) from params_array
        in suitable form by splitting it.

        Parameters:
            params_array: array of variational parameters
            no_layers: number of data-uploading layers

        Returns:
            lambda_array, amplitudes_array, phases_array, and w: subarrays of variational parameters in suitable form
        """

        # asserts
        assert no_layers > 0, "no_layers > 0 required"

        no_pos_freqs = no_layers + 1
        no_freqs = 2 * no_layers + 1

        assert len(params_array) == 2 + 2 * no_pos_freqs * no_freqs + 1, \
            f"length {len(params_array)} of params_array not correct; it must contain 2 input scaling parameters, " \
            "(self.no_layers + 1) * (2 * self.no_layers + 1) amplitudes, " \
            "(self.no_layers + 1) * (2 * self.no_layers + 1) phases, " \
            f"and 1 output scaling parameter, here in total: {2 + 2 * no_pos_freqs * no_freqs + 1}"

        # extract subarrays
        lambda_array = params_array[:2]
        coeffs_array = params_array[2:-1]
        w = params_array[-1:]

        no_pos_freqs = no_layers + 1
        no_freqs = 2 * no_layers + 1

        amplitudes_array = coeffs_array[:no_pos_freqs * no_freqs].reshape(no_pos_freqs, no_freqs)
        phases_array = coeffs_array[no_pos_freqs * no_freqs:].reshape(no_pos_freqs, no_freqs)

        return lambda_array, amplitudes_array, phases_array, w


    @staticmethod
    def get_params_array_from_subarrays(lambda_array: np.ndarray, amplitudes_array: np.ndarray,
                                        phases_array: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Get params_array from subarrays of variational parameters (lambda_array, amplitudes_array, phases_array, and w)
        in suitable form by concatenating them.

        Parameters:
            lambda_array: array of input scaling parameters
            amplitudes_array: array of amplitudes of Fourier coefficients
            phases_array: array of phases of Fourier coefficients
            w: weight/output scaling parameter

        Returns:
            params_array: array of variational parameters in suitable form
        """

        return np.concatenate((lambda_array, amplitudes_array.flatten(), phases_array.flatten(), w))


    @staticmethod
    def insert_optimized_params_into_larger_params_array(optimized_params_array: np.ndarray, params_array: np.ndarray,
                                                         optimized_no_layers: int, no_layers: int) -> np.ndarray:
        """
        Insert optimized parameters for parameterized quantum circuit (PQC) with optimized_no_layers-many data-uploading
        layers into params_array for PQC with no_layers-many data-uploading layers.

        Parameters:
            optimized_params_array: array of optimized variational parameters for PQC with optimized_no_layers-many
                                    data-uploading layers
            params_array: array of variational parameters for PQC with no_layers-many data-uploading layers
            optimized_no_layers: number of data-uploading layers for optimized PQC
            no_layers: number of data-uploading layers for PQC

        Returns:
            params_array: array of variational parameters for PQC with no_layers-many data-uploading layers containing
                          optimized parameters for PQC with optimized_no_layers-many data-uploading layers
        """

        # asserts
        assert optimized_no_layers > 0, "optimized_no_layers > 0 required"
        assert no_layers > 0, "no_layers > 0 required"

        opt_no_pos_freqs = optimized_no_layers + 1
        opt_no_freqs = 2 * optimized_no_layers + 1

        no_pos_freqs = no_layers + 1
        no_freqs = 2 * no_layers + 1

        assert len(optimized_no_layers) == 2 + 2 * opt_no_pos_freqs * opt_no_freqs + 1, \
            f"length {len(optimized_no_layers)} of optimized_no_layers not correct; it must contain "\
            "2 input scaling parameters, " \
            "(optimized_no_layers + 1) * (2 * optimized_no_layers + 1) amplitudes, " \
            "(optimized_no_layers + 1) * (2 * optimized_no_layers + 1) phases, " \
            f"and 1 output scaling parameter, here in total: {2 + 2 * opt_no_pos_freqs * opt_no_freqs + 1}"

        assert len(params_array) == 2 + 2 * no_pos_freqs * no_freqs + 1, \
            f"length {len(params_array)} of params_array not correct; it must contain 2 input scaling parameters, " \
            "(no_layers + 1) * (2 * no_layers + 1) amplitudes, " \
            "(no_layers + 1) * (2 * no_layers + 1) phases, " \
            f"and 1 output scaling parameter, here in total: {2 + 2 * no_pos_freqs * no_freqs + 1}"

        # initializations
        opt_lambda_array, opt_amplitudes_array, opt_phases_array, opt_w = \
            ParameterizedDynamicsFits.get_subarrays_from_params_array(optimized_params_array, optimized_no_layers)
        lambda_array, amplitudes_array, phases_array, w = \
            ParameterizedDynamicsFits.get_subarrays_from_params_array(params_array, no_layers)

        lambda_array = opt_lambda_array
        w = opt_w

        # insert optimized amplitudes and phases
        amplitudes_array[:(optimized_no_layers + 1),
                         (no_layers - optimized_no_layers):(no_layers + optimized_no_layers + 1)] = \
            opt_amplitudes_array

        phases_array[:(optimized_no_layers + 1),
                     (no_layers - optimized_no_layers):(no_layers + optimized_no_layers + 1)] = \
            opt_phases_array

        return ParameterizedDynamicsFits.get_params_array_from_subarrays(lambda_array, amplitudes_array, phases_array,
                                                                         w)


    @staticmethod
    def calc_params_array_n_qubits_1_layer(no_qubits: int, params_n_qubits_array: np.ndarray, in_terms_of_thetas=True) \
            -> np.ndarray:
        """
        Calculate elements of params_array (and assemble it) suitable for method softmax_policy_Fourier_coeffs from
        independent parameters of parameterized quantum circuits (PQC) with no_qubits-many qubits and
        1 data-uploading layer.

        Parameters:
            no_qubits: #qubits of PQC
            params_n_qubits_array: array of independent parameters of PQC
            in_terms_of_thetas: if True, variational angles theta (in params_n_qubits_array) are used to determine
                                amplitudes and phases in truncated Fourier series;
                                else: amplitudes and phases (in params_n_qubits_array) are directly used

        Returns:
            params_array: array of variational parameters suitable for method softmax_policy_Fourier_coeffs
        """

        # asserts
        assert no_qubits > 0, "no_qubits > 0 required"

        if no_qubits > 2:
            raise NotImplementedError("Case no_qubits > 2 has not been implemented yet.")

        if no_qubits == 1:
            if in_terms_of_thetas:
                assert len(params_n_qubits_array[2:-1]) == 3, \
                    "amplitudes and phases in truncated Fourier series are determined in the 1-qubit-1-layer case " \
                    "by 3 angles theta_1, theta_2, and theta_3, which must be supplied in params_array[2:-1]"
            else:
                assert len(params_n_qubits_array[2:-1]) == 6, \
                    "amplitudes and phases in truncated Fourier series are determined in the 1-qubit-1-layer case " \
                    "by 3 non-zero amplitudes and 3 non-zero phases, which must be supplied in this order in " \
                    "params_array[2:-1]"

        elif no_qubits == 2:
            if in_terms_of_thetas:
                assert len(params_n_qubits_array[2:-1]) == 2, \
                    "amplitudes and phases in truncated Fourier series are determined in the 2-qubits-1-layer case " \
                    "by 2 angles theta_1 and theta_2 which must be supplied in params_array[2:-1]"
            else:
                assert len(params_n_qubits_array[2:-1]) == 1, \
                    "amplitudes and phases in truncated Fourier series are determined in the 2-qubits-1-layer case " \
                    "by 1 non-zero amplitudes, which must be supplied in this order in params_array[2:-1]"

        # calculate elements of params_array suitable for method softmax_policy_Fourier_coeffs
        if in_terms_of_thetas:
            # complex-valued coefficients of truncated Fourier series
            c_array = np.zeros(2 * 3, dtype=complex)

            # the following Fourier coefficients are hard-coded into this method in order to achieve
            # a higher/the highest performance
            if no_qubits == 1:
                theta_1, theta_2, theta_3 = params_n_qubits_array[2:-1]

                c_array[1 * 3 + 0] = 1 / 4 * (
                            np.cos(theta_1) - np.cos(theta_2) - 1j * np.sin(theta_1) * np.sin(theta_2)) \
                                     * np.cos(theta_3)
                c_array[1 * 3 + 1] = - 1 / 2 * (np.sin(theta_1) * np.cos(theta_2) - 1j * np.sin(theta_2)) \
                                     * np.sin(theta_3)
                c_array[1 * 3 + 2] = 1 / 4 * (
                            np.cos(theta_1) + np.cos(theta_2) - 1j * np.sin(theta_1) * np.sin(theta_2)) \
                                     * np.cos(theta_3)

            elif no_qubits == 2:
                theta_1, theta_2 = params_n_qubits_array[2:-1]

                c_array[1 * 3 + 0] = 1 / 4 * np.cos(theta_1) * np.cos(theta_2)
                c_array[1 * 3 + 2] = 1 / 4 * np.cos(theta_1) * np.cos(theta_2)

            # corresponding amplitudes and phases in truncated Fourier series
            a_array = 2 * np.abs(c_array)
            phi_array = np.angle(c_array)

        else:
            a_array = np.zeros(2 * 3)
            phi_array = np.zeros(2 * 3)

            if no_qubits == 1:
                for i in range(3):
                    a_array[1 * 3 + i] = params_n_qubits_array[2 + i]
                    phi_array[1 * 3 + i] = params_n_qubits_array[2 + 3 + i]

            elif no_qubits == 2:
                a_array[1 * 3 + 0] = params_n_qubits_array[2]
                a_array[1 * 3 + 2] = params_n_qubits_array[2]

        # construct params_array suitable for method softmax_policy_Fourier_coeffs
        params_array = np.zeros(2 + len(a_array) + len(phi_array) + 1)
        params_array[:2] = params_n_qubits_array[:2]
        params_array[2:-1] = np.concatenate((a_array, phi_array))
        params_array[-1:] = params_n_qubits_array[-1:]

        return params_array


    @staticmethod
    def softmax_policy_function_from_sympy_expr(expectation_val: sp.Expr, no_layers: int) \
            -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """
        Calculate softmax policy function from SymPy expression for expectation value of observable for
        parameterized quantum circuit (PQC) with no_layers-many data-uploading layers and lambdify it.

        Parameters:
            expectation_val: SymPy expression for expectation value of observable
            no_layers: number of data-uploading layers of PQC

        Returns:
            lambdified softmax policy function
        """

        # asserts
        assert no_layers > 0, "no_layers > 0 required"

        # define variables (SymPy symbols)
        t, x = sp.symbols("t, x", real=True)
        lambda_t, lambda_x = sp.symbols("lambda_t, lambda_x", real=True)

        no_thetas = 4 * no_layers - 1
        thetas = list(sp.symbols("theta1:" + str(no_thetas + 1), real=True))

        w = sp.symbols("w", real=True)

        # substitute variables
        expectation_val = expectation_val.subs(t, sp.atan(lambda_t * t))
        expectation_val = expectation_val.subs(x, sp.atan(lambda_x * x))

        # apply softmax function
        softmax_policy = 1 / (sp.exp(w * expectation_val) + 1)

        # lambdify SymPy expression
        return sp.lambdify([t, x, lambda_t, lambda_x] + thetas + [w], softmax_policy)


    def softmax_policy_from_lambdified_expr(self, coords_array: np.ndarray, params_array: np.ndarray) -> np.ndarray:
        """
        Calculate softmax policy array for parameterized quantum circuit (PQC) via class attribute
        softmax_policy_lambdified computed via method softmax_policy_function_from_sympy_expr.

        Parameters:
            coords_array: array of coordinates (t, x) for which to calculate the policy
            params_array: array of variational parameters of PQC for which to calculate the policy

        Returns:
            softmax policy array for given coordinates and parameters
        """

        no_pos_freqs = self.no_layers + 1
        no_freqs = 2 * self.no_layers + 1

        assert len(params_array) == 2 + 2 * no_pos_freqs * no_freqs + 1, \
            f"length {len(params_array)} of params_array not correct; it must contain 2 input scaling parameters, " \
            "(self.no_layers + 1) * (2 * self.no_layers + 1) amplitudes, " \
            "(self.no_layers + 1) * (2 * self.no_layers + 1) phases, " \
            f"and 1 output scaling parameter, here in total: {2 + 2 * no_pos_freqs * no_freqs + 1}"

        no_t, no_x, _ = np.shape(coords_array)

        return np.array([[self.softmax_policy_lambdified(coords_array[m, n, 0], coords_array[m, n, 1], *params_array)
                          for n in range(no_x)]
                         for m in range(no_t)])


    def softmax_policy_Fourier_coeffs(self, coords_array: np.ndarray, params_array: np.ndarray) -> np.ndarray:
        """
        Calculate softmax policy array for parameterized quantum circuit (PQC) from Fourier coefficients.

        Parameters:
            coords_array: array of coordinates (t, x) for which to calculate the policy
            params_array: array of Fourier coefficients of PQC for which to calculate the policy

        Returns:
            softmax policy array for given coordinates and Fourier coefficients
        """

        no_pos_freqs = self.no_layers + 1
        no_freqs = 2 * self.no_layers + 1

        # asserts
        assert len(params_array) == 2 + 2 * no_pos_freqs * no_freqs + 1, \
            f"length {len(params_array)} of params_array not correct; it must contain 2 input scaling parameters, " \
            "(self.no_layers + 1) * (2 * self.no_layers + 1) amplitudes, " \
            "(self.no_layers + 1) * (2 * self.no_layers + 1) phases, " \
            f"and 1 output scaling parameter, here in total: {2 + 2 * no_pos_freqs * no_freqs + 1}"

        # initializations
        lambda_array, amplitudes_array, phases_array, w = self.get_subarrays_from_params_array(params_array,
                                                                                               self.no_layers)

        pos_freqs = np.arange(0, self.no_layers + 1)
        freqs = np.arange(- self.no_layers, self.no_layers + 1)
        freqs_t, freqs_x = np.meshgrid(pos_freqs, freqs, indexing="ij")
        # -> increasing first index corresponds to increasing frequency for oscillations in t,
        #    increasing second ----------------------------"---------------------------- in x
        # NOTE: due to the symmetry cos(-x) = cos(x), one does NOT have to consider negative frequencies for
        # either g_t or g_x; here g_t is chosen

        amplitudes_array = np.broadcast_to(amplitudes_array, (*np.shape(coords_array)[:-1], no_pos_freqs, no_freqs))
        phases_array = np.broadcast_to(phases_array, (*np.shape(coords_array)[:-1], no_pos_freqs, no_freqs))

        # scale input (coordinates)
        g_t = np.arctan(lambda_array[0] * coords_array[..., 0])
        g_x = np.arctan(lambda_array[1] * coords_array[..., 1])

        # calculate Fourier series representation
        series_representation = amplitudes_array * np.cos(np.einsum(einsum_subscripts("ft,fx",
                                                                                      "gt,gx",
                                                                                      final_indices="gt,gx,ft,fx"),
                                                                    freqs_t,
                                                                    g_t)
                                                          + np.einsum(einsum_subscripts("ft,fx",
                                                                                        "gt,gx",
                                                                                        final_indices="gt,gx,ft,fx"),
                                                                      freqs_x,
                                                                      g_x)
                                                          + phases_array)

        # calculate softmax policy
        return 1 / (np.exp(w * np.einsum(einsum_subscripts("gt,gx,ft,fx",
                                                           final_indices="gt,gx"),
                                         series_representation))
                    + 1)


    def fit_policy(self, policy: Callable[[np.ndarray, np.ndarray], np.ndarray], coords_array: np.ndarray,
                   optimal_policy_array: np.ndarray, no_qubits: int, no_layers: int, no_fits: int, cost_func_type: str,
                   no_trajectories_cost_func: int = None, T: int = None, s: float = None, x_T: float = None,
                   prob_step_up: float = None, optimal_average_return: float = None, no_thetas: int = None,
                   no_amplitudes: int = None, no_phases: int = None, optimized_params: np.ndarray = None,
                   optimized_no_layers: int = None, compute_in_parallel=False, max_optimization_steps: int = None) \
            -> tuple[np.ndarray, np.ndarray]:
        """
        Fit policy to optimal policy by calling method fit_policy_parallelizable, calculate function values arrays for
        optimized parameters, and search for optimized parameters with minimal residual cost.

        Parameters:
            policy: policy (as function of fitting parameters) to be fitted to optimal policy
            coords_array: array of coordinates (t, x) for which the policy is to be fitted to the optimal policy
            optimal_policy_array: array of optimal policy values for the coordinates in coords_array
            no_qubits: #qubits of parameterized quantum circuit (PQC)
            no_layers: #data-uploading layers of PQC
            no_fits: #fits to be performed (each with independent and identically distributed initial guesses for
                                            the fitting parameters)
            cost_func_type: type of cost function to be minimized
                            (currently cost_func_type == "leastsq" and "trajectory_KL_divergence" implemented)
            no_trajectories_cost_func: #trajectories for cost_func_type == "trajectory_KL_divergence"
            T: #time steps of random walk/trajectories for cost_func_type == "trajectory_KL_divergence"
            s: parameter in reward of trajectories for cost_func_type == "trajectory_KL_divergence"
            x_T: required end point of trajectories for cost_func_type == "trajectory_KL_divergence"
            prob_step_up: probability of 1 step up in original random walk, required to calculate rewards for
                          cost_func_type == "trajectory_KL_divergence"
            optimal_average_return: average return of optimal policy for cost_func_type == "trajectory_KL_divergence"
            no_thetas: #variational parameters thetas of policy
            no_amplitudes: #amplitudes of Fourier coefficients of policy
            no_phases: #phases of Fourier coefficients of policy
            optimized_params: initial guess for fitting parameters already optimized in the case of PQC
                                             with optimized_no_layers-many data-uploading layers
            optimized_no_layers: see optimized_params_fourier_coeffs
            compute_in_parallel: if True, the fits are computed in parallel
            max_optimization_steps: maximum number of optimization steps for each fit

        Returns:
            fitted_policies_array: array of function values for optimized parameters
            optimized_params_min: optimized parameters with minimal residual cost
        """

        # asserts
        if cost_func_type == "trajectory_KL_divergence":
            assert no_trajectories_cost_func is not None, \
                'if cost_func_type == "trajectory_KL_divergence", no_trajectories_cost_func must be provided'
            assert no_trajectories_cost_func > 0, \
                'no_trajectories_cost_func > 0 required'
            assert T is not None, 'T must be provided'
            assert s is not None, 's must be provided'
            assert x_T is not None, 'x_T must be provided'
            assert prob_step_up is not None, 'prob_step_up must be provided'
            assert optimal_average_return is not None, 'optimal_average_return must be provided'

        assert (no_thetas is not None) ^ (no_amplitudes is not None and no_phases is not None), \
            'either no_thetas or (no_amplitudes and no_phases) must be provided'

        # initializations
        optimized_params_list = []
        residual_cost_list = []

        # initialize instance progress_bar of utility class ProgressBar
        progress_bar = ProgressBar(no_fits, f"Fits for #qubits: {no_qubits}, #layers: {no_layers}")

        single_job_params = [policy, coords_array, optimal_policy_array, no_layers, cost_func_type,
                             no_trajectories_cost_func, T, s, x_T, prob_step_up, optimal_average_return, no_thetas,
                             no_amplitudes, no_phases, optimized_params, optimized_no_layers,
                             max_optimization_steps]
        # it remains to append #fits (to be done by single job) to single_job_params

        if compute_in_parallel:
            # determine job sizes
            cpu_count = os.cpu_count()
            job_size = no_fits // (cpu_count - 1)
            rest_size = no_fits % (cpu_count - 1)
            job_params = []

            for cpu_index in range(cpu_count):
                # compute #fits to be done by single job
                single_job_no_fits = job_size if cpu_index < (cpu_count - 1) else rest_size

                # append #fits (to be done by single job) to single_job_params
                # and append single_job_params to job_params
                job_params.append(single_job_params + [single_job_no_fits])

            # serialize job_params
            serialized_job_params = [dill.dumps(params, recurse=True) for params in job_params]

            progress_bar.update()

            # compute fits in parallel
            with Pool() as pool:
                for task_result in pool.imap_unordered(ParameterizedDynamicsFits.fit_policy_in_parallel,
                                                       serialized_job_params):

                    optimized_params_sublist, residual_cost_sublist = task_result
                    optimized_params_list += optimized_params_sublist
                    residual_cost_list += residual_cost_sublist

                    progress_bar.update(new_current_step=progress_bar.current_step + len(task_result))

        else:
            # compute fits in sequence
            single_job_params.append(1)  # append #fits = 1 (to be done by single job) to single_job_params

            for i in range(no_fits):
                # update progress_bar due to progress
                progress_bar.update(i)

                optimized_params_sublist, residual_cost_sublist = \
                    ParameterizedDynamicsFits.fit_policy_parallelizable(single_job_params)

                optimized_params_list += optimized_params_sublist
                residual_cost_list += residual_cost_sublist

        # finish progress bar
        progress_bar.finish()

        # calculate function values arrays for optimized parameters
        fitted_policies_array = np.zeros((len(optimized_params_list), *np.shape(optimal_policy_array)))
        optimized_params_min = optimized_params_list[0]
        min_residual_cost = residual_cost_list[0]

        for n, params in enumerate(optimized_params_list):
            fitted_policies_array[n] = policy(coords_array, params)

            # search for optimized parameters with minimal residual cost
            if residual_cost_list[n] < min_residual_cost:
                optimized_params_min = params

        return fitted_policies_array, optimized_params_min


    @staticmethod
    def fit_policy_in_parallel(serialized_job_params: list) -> tuple[list, list]:
        """
        Helper method for serialization of functions generated by sympy.lambdify needed in parallel computation
        via pool.imap_unordered in method fit_policy.

        Parameters:
            serialized_job_params: see description of job_params in method fit_policy_parallelizable

        Returns:
            optimized_params_sublist: list of optimized parameters for policy
            residual_cost_sublist: list of residual costs for policy
        """

        # deserialize job_params
        job_params = dill.loads(serialized_job_params)

        return ParameterizedDynamicsFits.fit_policy_parallelizable(job_params)


    @staticmethod
    def fit_policy_parallelizable(job_params: list) -> tuple[list, list]:
        """
        Fit policy to optimal policy in parallelizable way (after choosing initial guesses for fitting parameters and
        defining their bounds). To be used in combination with method fit_policy.

        Parameters:
            job_params: list of parameters for fitting policy to optimal policy
                        (policy, coords_array, optimal_policy_array, no_layers, cost_func_type,
                         no_trajectories_cost_func, T, s, x_T, prob_step_up, optimal_average_return, no_thetas,
                         no_amplitudes, no_phases, optimized_params_fourier_coeffs, optimized_no_layers,
                         max_optimization_steps, no_fits)

        Returns:
            optimized_params_sublist: list of optimized parameters for policy
            residual_cost_sublist: list of residual costs for policy
        """

        # initializations
        (policy, coords_array, optimal_policy_array, no_layers, cost_func_type, no_trajectories_cost_func,
         T, s, x_T, prob_step_up, optimal_average_return, no_thetas, no_amplitudes, no_phases,
         optimized_params_fourier_coeffs, optimized_no_layers, max_optimization_steps, no_fits) = \
            job_params

        optimized_params_sublist = []
        residual_cost_sublist = []

        for fit in range(no_fits):
            if no_thetas is not None:
                # sample random initial guess for fitting parameters
                initial_scalings = np.random.standard_normal(3)
                initial_thetas = 2 * np.pi * np.random.random(no_thetas)
                initial_params = np.insert(initial_scalings, 2, initial_thetas)
                # inserts initial_thetas into initial_scalings starting at position 2

                # define bounds for fitting parameters
                bounds_params = ([(-np.inf, np.inf)] * 2
                                 + [(0., 2 * np.pi)] * no_thetas
                                 + [(-np.inf, np.inf)])

            if no_amplitudes is not None and no_phases is not None:
                if optimized_params_fourier_coeffs is None or optimized_no_layers is None:
                    # sample random initial guess for fitting parameters
                    initial_scalings = np.random.standard_normal(3 + no_amplitudes)
                    initial_phases = 2 * np.pi * np.random.random(no_phases)
                    initial_params = np.insert(initial_scalings, 2 + no_amplitudes, initial_phases)
                    # inserts initial_phases into initial_scalings starting at position 2 + no_amplitudes

                else:
                    # use optimized_params_fourier_coeffs as initial guess for as much fitting parameters as possible
                    # and sample random initial guess for the fitting parameters left
                    initial_params = np.zeros(2 + no_amplitudes + no_phases + 1)
                    initial_params = \
                        ParameterizedDynamicsFits.insert_optimized_params_into_larger_params_array(
                            optimized_params_fourier_coeffs,
                            initial_params,
                            optimized_no_layers,
                            no_layers)

                # define bounds for fitting parameters
                bounds_params = ([(-np.inf, np.inf)] * (2 + no_amplitudes)
                                 + [(0., 2 * np.pi)] * no_phases
                                 + [(-np.inf, np.inf)])

            # fit policy to optimal policy
            optimized_params, residual_cost = \
                ParameterizedDynamicsFits.fit_func_to_data(policy, coords_array, optimal_policy_array,
                                                           params_initial_guess=initial_params,
                                                           params_bounds=bounds_params, no_independent_vars=2,
                                                           cost_func_type=cost_func_type,
                                                           no_trajectories_cost_func=no_trajectories_cost_func,
                                                           T=T, s=s, x_T=x_T, prob_step_up=prob_step_up,
                                                           average_return_data=optimal_average_return,
                                                           max_optimization_steps=max_optimization_steps)

            optimized_params_sublist.append(optimized_params)
            residual_cost_sublist.append(residual_cost)

        return optimized_params_sublist, residual_cost_sublist


    @staticmethod
    def fit_func_to_data(func: Callable[[np.ndarray, np.ndarray], np.ndarray], coords_array: np.ndarray,
                         data_array: np.ndarray, params_initial_guess: np.ndarray, params_bounds=None,
                         asserts=True, no_independent_vars: int = None, cost_func_type="leastsq",
                         no_trajectories_cost_func: int = None, T: int = None, s: float = None, x_T: int = None,
                         prob_step_up: float = None, average_return_data: float = None,
                         max_optimization_steps: int = None) -> tuple[np.ndarray, float]:
        """
        Fit a function to given data by minimizing a cost function.

        Parameters:
            func: function to be fitted to the data
            coords_array: coordinates of the data points
            data_array: data points
            params_initial_guess: initial guess for the fitting parameters of the function
            params_bounds: bounds for the fitting parameters
            asserts: if True, asserts are checked
            no_independent_vars: number of independent variables of the function (only needed if asserts == True)
            cost_func_type: type of cost function to be minimized
                (currently cost_func_type == "leastsq" and "trajectory_KL_divergence" implemented)
            no_trajectories_cost_func: #trajectories for cost_func_type == "trajectory_KL_divergence"
            T: #time steps of random walk/trajectories for cost_func_type == "trajectory_KL_divergence"
            s: parameter in reward of trajectories for cost_func_type == "trajectory_KL_divergence"
            x_T: required end point of trajectories for cost_func_type == "trajectory_KL_divergence"
            prob_step_up: probability of 1 step up in original random walk, required to calculate rewards for
                          cost_func_type == "trajectory_KL_divergence"
            average_return_data: average return of data_array for cost_func_type == "trajectory_KL_divergence"
            max_optimization_steps: maximum number of optimization steps

        Returns:
            optimized_params: optimized fitting parameters
            residual_cost: residual value of cost function after fitting
        """

        # asserts
        if asserts:
            assert no_independent_vars is not None, \
                "If asserts == True, no_independent_vars must be specified!"

            shape_coords_array = np.shape(coords_array)
            shape_data_array = np.shape(data_array)

            assert shape_coords_array[-1] == no_independent_vars, \
                "Shape of coords_array[-1] must equal no_independent_vars!"
            assert shape_coords_array[:-1] == shape_data_array, \
                "Shape of coords_array[:-1] must equal shape of data_array!"

            func_vals_array = func(coords_array, params_initial_guess)

            shape_func_vals_array = np.shape(func_vals_array)

            assert shape_func_vals_array == shape_data_array, \
                "Shape of func(coords_array, params_initial_guess) must equal shape of data_array!"

        if cost_func_type == "trajectory_KL_divergence":
            assert no_trajectories_cost_func is not None, \
                'if cost_func_type == "trajectory_KL_divergence", no_trajectories must be provided'
            assert T is not None, 'if cost_func_type == "trajectory_KL_divergence", T must be provided'
            assert s is not None, 'if cost_func_type == "trajectory_KL_divergence", s must be provided'
            assert x_T is not None, 'if cost_func_type == "trajectory_KL_divergence", x_T must be provided'
            assert prob_step_up is not None, \
                'if cost_func_type == "trajectory_KL_divergence", prob_step_up must be provided'
            assert average_return_data is not None, \
                'if cost_func_type == "trajectory_KL_divergence", average_return_data must be provided'

        # define cost function
        if cost_func_type == "leastsq":
            def cost_func(params):
                # calculate values of func for coords_array and params
                func_vals_array = func(coords_array, params)

                # calculate and return mean squared error as cost, ignoring NaNs
                return np.real(np.nanmean((func_vals_array - data_array) ** 2))

        elif cost_func_type == "trajectory_KL_divergence":
            def cost_func(params):
                # calculate values of func for coords_array and params
                func_vals_array = func(coords_array, params)

                # generate trajectories for P_theta
                policies_array = np.expand_dims(func_vals_array, 0),  # to add dimension for different policies

                trajectories_x_array = \
                    PolicyEvaluation.calc_trajectories_x_array(policies_array, T, no_trajectories_cost_func)

                # calculate return values and estimate average return for P_theta
                return_values = PolicyEvaluation.calc_return_values(trajectories_x_array, policies_array, s, x_T,
                                                                    prob_step_up)

                average_return_estimate = np.mean(return_values, axis=-1)[0]

                # calculate Kullback-Leibler divergence
                estimate_KL_divergence = average_return_data - average_return_estimate

                if estimate_KL_divergence < 0:
                    raise ValueError("KL divergence estimate is negative, so no_trajectories is chosen too small")
                else:
                    return np.real(estimate_KL_divergence)

        else:
            raise NotImplementedError(f'Option cost_func_type="{cost_func_type}" has not been implemented yet.')

        # fit by minimizing cost function
        if max_optimization_steps is None:
            result = minimize(cost_func, params_initial_guess, bounds=params_bounds)

        else:
            result = minimize(cost_func, params_initial_guess, bounds=params_bounds, method='L-BFGS-B',
                              options={'maxiter': max_optimization_steps})

        if not result.success:
            logger.debug("Fitting was not successful according to the default success criterion of "
                         "scipy.optimize.minimize.")

        optimized_params = result.x
        residual_cost = result.fun

        return optimized_params, residual_cost



