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


import time
import json5
import numpy as np
import _pickle
import sympy as sp
from abc import ABC, abstractmethod
from logging_config import get_logger


logger = get_logger("utilities.py")


def import_params_from_json5(path: str) -> dict:
    """
    Import parameters from a JSON5 file and return them as a dictionary.

    Parameters:
        path: path to the JSON5 file

    Returns:
        params: dict with parameter names and values as dict key-value pairs
    """
    with open(path, 'r') as file:
        params = json5.load(file)
    return params


def convert_to_and_save_latex_string(sympy_expr: sp.Expr, file_name: str, comment: str) -> None:
    """
    Convert sympy expression to equivalent LaTeX string and save it in txt file.

    Parameters:
        sympy_expr: sympy expression to be converted
        file_name: name of the file to save the LaTeX string
        comment: comment to be saved in the last line of the file

    Returns:
        None
    """

    latex_string = sp.latex(sympy_expr, mode='inline')

    with open(file_name, 'w') as file:
        file.write(latex_string + '\n')
        file.write(comment + '\n')


def einsum_subscripts(*initial_indices: str, final_indices="") -> str:
    """
    Convert strings of indices in *initial_indices and string of indices in **final_indices to one string suitable as
    parameter "subscript" of numpy.einsum (see its documentation for details). (The strings can contain numbers, primes,
    superscripts, etc. like in common physicists' notation, unlike numpy.einsum subscripts.)

    Parameters:
        *initial_indices: strings of indices for operands of numpy.einsum
        final_indices: string of indices for result of numpy.einsum

    Returns:
        translated_indices: string of indices suitable as parameter "subscript" of numpy.einsum
    """

    # initializations
    letters_list = list(map(chr, range(97, 123)))  # list of all letters in the alphabet
    indices_dict = {}  # dictionary which contains to correspondences between subscripts in SB21/my notes
                         # and subscripts conforming with the conventions/requirements of np.einsum
    initial_indices_lists = [index.split(",") for index in initial_indices]
    final_indices_list = final_indices.split(",")
    translated_indices = ""

    # utility function
    def translate_and_add(index, translated_indices, indices_dict):
        # translate index according to indices_dict and add it to string translated_indices
        if index == "..." or index == "":
            translated_indices += index

        else:
            translated_indices += indices_dict[index]

        return translated_indices

    # translate indices
    for indices_list in initial_indices_lists:
        for index in indices_list:
            if index not in indices_dict.keys():
                # add index to indices_dict
                l = len(indices_dict)
                indices_dict.update({index: letters_list[l]})

            translated_indices = translate_and_add(index, translated_indices, indices_dict)

        translated_indices += ","  # in np.einsum, indices of different operands are to be separated by commas

    translated_indices = translated_indices[:-1]  # remove last comma
    translated_indices += "->"

    for index in final_indices_list:
        translated_indices = translate_and_add(index, translated_indices, indices_dict)

    return translated_indices


def save_obj(obj, file_name: str) -> None:
    """
    Save all attributes of instance obj of a class to a file with name file_name.

    Parameters:
        obj: instance of a class
        file_name: string ending with '.npz'

    Returns:
        None
    """

    attributes = obj.__dict__

    try:
        np.savez(file_name, **attributes)

    except _pickle.PicklingError:
        logger.info(f"Object cannot be saved in {file_name} due to PicklingError.")


def load_and_restore_obj(class_type, file_name: str, all_params_dict: dict):
    """
    Load all attributes of an instance of class cls from a file with name file_name,
    create a new instance of class cls, and update its attributes accordingly.

    Parameters:
        class_type: class
        file_name: string ending with '.npz'

    Returns:
        obj: instance of class class_type
    """

    data = np.load(file_name, allow_pickle=True)

    obj = class_type.__new__(class_type)

    attributes = {f: data[f] for f in data.files}

    # combination of np.savez in function save_obj and np.load above leads to attributes originally of type
    # 'int', 'float', 'str', 'bool', etc. being loaded as np.ndarray objects with a single element
    # -> revert this
    for key, value in attributes.items():
        if value.size == 1:
            attributes[key] = value.item()

    obj.__dict__.update(attributes)

    if not isinstance(obj, ConsistentParametersClass):
        error_message = (f"Class {class_type} must inherit from class ConsistentParametersClass to ensure that its "
                         f"objects can be restored with consistent parameters.")

        logger.debug(error_message)
        raise AssertionError(error_message)

    if not isinstance(obj.all_init_params_dict, dict):
        error_message = (f"Property all_init_params_dict of class {class_type} must be a dictionary of all "
                         f"parameter-value pairs used to initialize an instance.")

        logger.debug(error_message)
        raise AssertionError(error_message)

    parameters_are_not_consistent = False

    for key, value in obj.all_init_params_dict.items():
        if key == "no_qubits":
            if not (value in all_params_dict["no_qubits_list"]):
                print(key, value)
                parameters_are_not_consistent = True
                break

        elif key == "no_layers":
            if not (value in all_params_dict["no_layers_list"]):
                print(key, value)
                parameters_are_not_consistent = True
                break

        elif not key in all_params_dict.keys():
            print(key, value, all_params_dict[key])
            parameters_are_not_consistent = True
            break

        elif all_params_dict[key] != value:
            print(key, value, all_params_dict[key])
            parameters_are_not_consistent = True
            break

    if parameters_are_not_consistent:
        error_message = ("All parameter-value pairs in all_init_params_dict of loaded object must be consistent with "
                         "parameter-value pairs in all_params_dict which contains all parameter-value pairs used for "
                         "current computations.")

        logger.debug(error_message)
        logger.debug(f"all_init_params_dict of loaded object: {obj.all_init_params_dict}")
        logger.debug(f"all_params_dict used for current computations: {all_params_dict}")
        raise AssertionError(error_message)

    return obj


def load_or_compute_obj(class_type, generator_function, file_name: str, all_params_dict: dict, load=True,
                        recompute=False):
    """
    Load and restore an object of class class_type from a file file_name if it exists,
    otherwise compute it and save it to the file.

    Parameters:
        class_type: class
        generator_function: function that generates the object
        file_name: string ending with '.npz'
        load: if True, load the object from the file if it exists
        recompute: if True, recompute the object even if it exists in the file and overwrite the file

    Returns:
        obj: instance of class class_type
    """

    if not recompute:
        try:
            obj = load_and_restore_obj(class_type, file_name, all_params_dict)

            if load:
                logger.info(f"Loaded object of class {class_type} from {file_name}.")
                return obj

            else:
                logger.info(f"Object of class {class_type} in {file_name} exists, but not loaded.")
                return

        except FileNotFoundError:
            logger.info(f"File {file_name} not found.")

        except AssertionError:
            logger.info(f"Consistent restoring of loaded object of {class_type} from {file_name} is not possible.")

        except _pickle.PicklingError:
            logger.info(f"Object of {class_type} from {file_name} cannot be loaded due to PicklingError.")

    # so if recompute is True or if load_and_restore_obj failed due to file with file_name not existing
    # or due to all_init_params_dict of loaded object not consistent with all_params_dict used for current computations:
    # recompute the object and save it to the file

    obj = generator_function()
    logger.info(f"Computed object of {class_type}.")

    save_obj(obj, file_name)
    logger.info(f"Saved object of {class_type} in {file_name}.")

    return obj


class ConsistentParametersClass(ABC):
    """
    Abstract base class for classes for which one wants to use function load_and_restore_obj (for example via function
    load_or_compute_obj) to load and restore instances of the respective class from files. The combination of this class
    and this function ensures that the parameters which had been used to initialize the now restored instance are
    consistent with the parameters which are used in the current computations.
    """
    def __init__(self):
        pass

    @property
    @abstractmethod
    def all_init_params_dict(self):
        """Dictionary of all parameter-value pairs used to initialize the respective object."""
        pass


class ProgressBar:
    """
    Class to create and update a progress bar for a task with a given number of steps.
    """

    def __init__(self, total_no_steps: int | None = None, task_description: str = ""):
        """
        Initialize progress bar by setting attributes.

        Parameters:
            total_no_steps: total number of steps
            task_description: description of the task to be performed
        """

        self.start_time = time.time()
        self.current_step = -1
        self.total_no_steps = total_no_steps
        self.task_description = task_description


    @staticmethod
    def format_delta_time(time_in_seconds: float) -> str:
        """
        Format time in seconds as hours, minutes, and seconds.

        Parameters:
            time_in_seconds: time in seconds

        Returns:
            formatted time string
        """

        seconds = time_in_seconds % 60
        minutes = int((time_in_seconds // 60) % 60)
        hours = int((time_in_seconds // 60 // 24))

        if hours:
            return f"{hours} h {minutes} min"
        elif minutes:
            return f"{minutes} min {seconds:0.1f} s"
        else:
            return f"{seconds:0.1f} s"


    def update(self, new_current_step: int | None = None, new_total_no_steps: int | None = None):
        """
        Update and print updated progress bar.

        Parameters:
            new_current_step: new current step (if None, increment current_step by 1)
            new_total_no_steps: new total number of steps (if None, keep total_no_steps)

        Returns:
            None
        """

        # update total_no_steps if necessary
        if new_total_no_steps is not None:
            self.total_no_steps = new_total_no_steps

        # update current_step
        self.current_step = new_current_step if new_current_step is not None else self.current_step + 1

        # calculate progress and expected time
        progress = self.current_step / self.total_no_steps

        elapsed_time = time.time() - self.start_time
        expected_time = elapsed_time / progress * (1 - progress) if progress else 0

        progress_bar_length = 20
        progress_done = int(progress * progress_bar_length)
        progress_bar_string = '=' * progress_done + ' ' * (progress_bar_length - progress_done)

        remaining_time_string = ProgressBar.format_delta_time(expected_time) if expected_time != 0 else "?"

        # print progress bar
        print(f"\x1b[1K\r  {self.task_description}: {self.current_step}/{self.total_no_steps} [{progress_bar_string}] "
              f"{progress * 100 : .0f} %, eta {remaining_time_string}", end="", flush=True)


    def finish(self):
        """
        Finish progress bar.

        Returns:
            None
        """

        # clear line
        print(f"\x1b[1K\r  {self.task_description} [{ProgressBar.format_delta_time(time.time() - self.start_time)}]")
