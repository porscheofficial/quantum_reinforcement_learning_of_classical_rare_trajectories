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
from typing import Literal

import json5
from pydantic import BaseModel, field_validator, ValidationInfo, model_validator
from pathlib import Path


class Config(BaseModel):
    T: int
    s: float
    x_T: int
    prob_step_up: float
    no_qubits_list: list[int]
    no_samples_variational_params: int
    no_layers_list: list[int]
    no_fits: int
    fitting_parameters: Literal["Fourier_coefficients", "variational_angles", "random_Fourier_features"]
    no_random_Fourier_features: int | None = None
    no_choices_random_Fourier_features: int | None = None
    max_optimization_steps: int | None = None
    cost_func_type: Literal["leastsq", "KL_divergence"]
    no_trajectories_cost_func: int | None = None
    no_trajectories_policy_evaluation: int
    policy_selection_criterion: Literal["max_prob_rare_trajectory", "max_avg_return", "min_KL_divergence", "min_MSE"]
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