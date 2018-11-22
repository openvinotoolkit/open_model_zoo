"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import os
import sys
import subprocess
from pathlib import Path
from typing import Union
from ..utils import check_exists

DEFAULT_MO_PATH = ('intel', 'computer_vision_sdk', 'deployment_tools', 'model_optimizer')

DEFAULT_PATHS = [
    Path.home().joinpath(*DEFAULT_MO_PATH).as_posix(),
    Path('/opt').joinpath(*DEFAULT_MO_PATH).as_posix(),
]


def convert_model(topology_name, output_directory: Path, model=None, weights=None,
                  framework='caffe', mo_search_paths=None, mo_params=None):
    """
    Args:
        topology_name: name for converted model files
        output_directory: directory for writing result
        model: path to the topology file
        weights: path to the weights file
        framework: framework name for original model
        mo_search_paths: paths where ModelOptimizer may be found. If None only default paths is used
        mo_params: parameters for ModelOptimizer execution
    Returns:
        paths to converted to IE IR model and weights
    """
    set_path_to_custom_operation_configs(mo_params, framework, output_directory)

    mo_params = prepare_mo_params(mo_params)

    model_optimizer_executable = find_mo(mo_search_paths)

    if model_optimizer_executable is None:
        raise EnvironmentError(
            "Model optimizer not found. Please set MO_DIR environment variable to model optimizer folder "
            "installation or refer to help for command line options for providing Model optimizer"
        )

    args = {
        '--model_name': topology_name,
        '--output_dir': output_directory.as_posix(),
        '--framework': framework
    }
    framework_specific_options = {'caffe': {'--input_model': weights.as_posix() if weights is not None else None,
                                            '--input_proto': model.as_posix() if model is not None else None},
                                  'mxnet': {'--input_model': weights.as_posix() if weights is not None else None},
                                  'tf': {'--input_model': model.as_posix() if model is not None else None},
                                  'onnx': {'--input_model': model.as_posix() if model is not None else None},
                                  'kaldi': {'--input_model': model.as_posix() if model is not None else None}}
    args.update(framework_specific_options.get(framework, {}))

    args.update(mo_params)
    args = prepare_args(model_optimizer_executable, value_options=args)
    code = exec_mo_binary(args)

    if code.returncode != 0:
        raise RuntimeError("Model optimizer conversion failed: ModelOptimizer returned non-zero code")

    model_file, bin_file = find_dlsdk_ir(output_directory.as_posix())

    if bin_file is None or model_file is None:
        raise RuntimeError("Model optimizer finished correctly, but converted model is not found.")

    return model_file, bin_file


def find_dlsdk_ir(search_path: str):
    """
    Args:
        search_path: path with IE IR of model
    Returns:
        paths to IE IR of model or None if they weren't found
    """
    for prefix, _, files in os.walk(search_path):
        xml_files = [f for f in files if f.endswith(".xml")]
        bin_files = [f for f in files if f.endswith(".bin")]
        if xml_files and bin_files:
            return (os.path.join(prefix, xml_files[0]),
                    os.path.join(prefix, bin_files[0]))
    return None, None


def find_mo(search_paths=None) -> Union[str, None]:
    """
    Args:
        search_paths: paths where ModelOptimizer may be found. If None only default paths is used
    Returns:
        path to the ModelOptimizer or None if it wasn't found
    """
    search_paths = search_paths or DEFAULT_PATHS

    executable = 'mo.py'

    for path in search_paths:
        path = Path(path)
        if not path.is_dir():
            continue

        mo = path / executable
        if not mo.is_file():
            continue

        return mo.as_posix()

    return None


def prepare_args(executable, flag_options=None, value_options=None):
    """
    Args:
        executable: path to the executable
        flag_options: positional arguments for executable
        value_options: keyword arguments for executable
    Returns:
        list with command-line entries
    """
    res = [sys.executable, executable]

    if flag_options:
        for arg in flag_options:
            res.append(str(arg))
    if value_options:
        for k, v in value_options.items():
            res.append(str(k))
            res.append(str(v))

    return res


def exec_mo_binary(args, timeout=None):
    """
    Args:
        args: command-line entries
        timeout: timeout for execution
    Returns:
        result of execution
    """
    return subprocess.run(args, check=False, timeout=timeout)


def prepare_mo_params(mo_params=None):
    if mo_params is None:
        return {}
    updated_mo_params = {}
    formated_key = '--{}'
    for k, v in mo_params.items():
        updated_mo_params[formated_key.format(k)] = v
    return updated_mo_params


def set_path_to_custom_operation_configs(mo_params, framework, output_directory):
    if framework != 'tf':
        return mo_params
    config_path = mo_params.get('tensorflow_use_custom_operations_config')

    if not config_path:
        return mo_params

    config_path = Path(config_path)
    config_path = output_directory / config_path
    check_exists(config_path.as_posix())
    mo_params['tensorflow_use_custom_operations_config'] = config_path
    return mo_params
