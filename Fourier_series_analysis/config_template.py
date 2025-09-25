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


import copy
import json5
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, field_validator, ValidationInfo, model_validator


class Config(BaseModel):
    # time steps of random walk
    T: int
    # reward/return function parameter
    s: float
    # required final position of rare trajectory
    x_T: int
    # random walk prob. to move one step up
    prob_step_up: float
    # list of #qubits in the parameterized quantum circuit (PQC)
    no_qubits_list: list[int]
    # #sets of randomly chosen variational angles = #times Fourier coefficients are computed
    no_samples_variational_params: int
    # list of #data-uploading layers in the PQC
    no_layers_list: list[int]
    # #sets of randomly chosen initial values for fitting parameters = #times policies/parameterized probs. are fitted to reweighted probs. P_W
    no_fits: int
    # fitting parameters to be used
    fitting_parameters: Literal["Fourier_coefficients", "variational_angles", "random_Fourier_features"]
    # #random Fourier features to use for fitting
    no_random_Fourier_features: int | None = None
    # (no comment in JSON5 for this field)
    no_choices_random_Fourier_features: int | None = None
    # maximal #optimization steps; if None, use default stopping criterion for optimization with scipy.optimize.minimize
    max_optimization_steps: int | None = None
    # type of cost function to use for fitting
    cost_func_type: Literal["leastsq", "KL_divergence"]
    # #trajectories used for cost_func_type='trajectory_KL_divergence'
    no_trajectories_cost_func: int | None = None
    # #trajectories used for estimating properties of fitted policies
    no_trajectories_policy_evaluation: int
    # criterion to select policy from no_fits many fits
    policy_selection_criterion: Literal["max_prob_rare_trajectory", "max_avg_return", "min_KL_divergence", "min_MSE"]
    # if True, recompute previous stored results
    recompute_stored: bool = False


    @classmethod
    @field_validator("T", "s", "no_samples_variational_params",
                     "no_random_Fourier_features", "no_choices_random_Fourier_features", "max_optimization_steps",
                     "no_trajectories_cost_func", "no_trajectories_policy_evaluation")
    def must_be_positive(cls, v, info: ValidationInfo):
        if v is None:
            return v

        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return v


    @classmethod
    @field_validator("prob_step_up")
    def must_be_probability(cls, v):
        if v <= 0:
            raise ValueError("prob_step_up must be between 0 and 1")
        return v


    @classmethod
    @field_validator("no_qubits_list", "no_layers_list")
    def must_be_list_of_positives(cls, v, info: ValidationInfo):
        if v is None:
            return v

        for num in v:
            num_info = copy.deepcopy(info)
            num_info.field_name += f"[{v.index(num)}]"
            Config.must_be_positive(num, num_info)
        return v


    @model_validator(mode="after")
    def must_of_same_parity(self):
        if not self.T % 2 == self.x_T % 2:
            raise ValueError("parameters T and x_T must have the same parity (both even or both odd)")
        return self


    @model_validator(mode="after")
    def must_not_be_none_if_random_Fourier_features(self):
        if self.fitting_parameters == "random_Fourier_features":
            if self.no_random_Fourier_features is None:
                raise ValueError("no_random_Fourier_features must be set if fitting_parameters is 'random_Fourier_features'")
            if self.no_choices_random_Fourier_features is None:
                raise ValueError("no_choices_random_Fourier_features must be set if fitting_parameters is 'random_Fourier_features'")
        return self


    @model_validator(mode="after")
    def must_not_be_none_if_KL_divergence(self):
        if self.cost_func_type == "KL_divergence":
            if self.no_trajectories_cost_func is None:
                raise ValueError("no_trajectories_cost_func must be set if cost_func_type is 'KL_divergence'")
        return self


    @classmethod
    def from_json5(cls, file_path: str | Path):
        """
        Load configuration from JSON5 file.
        """
        with open(file_path, "r") as f:
            params = json5.load(f)
        return cls(**params)