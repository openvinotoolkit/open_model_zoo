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

import sys
import subprocess
from pathlib import Path
from typing import Union
from ..utils import get_path, format_key

DEFAULT_MO_PATH = ('intel', 'computer_vision_sdk', 'deployment_tools', 'model_optimizer')

DEFAULT_PATHS = [
    Path.home().joinpath(*DEFAULT_MO_PATH),
    Path('/opt').joinpath(*DEFAULT_MO_PATH),
]


def convert_model(topology_name, output_directory: Path, model=None, weights=None,
                  framework='caffe', mo_search_paths=None, mo_params=None, mo_flags=None, tf_config_dir=None):
    """
    Args:
        topology_name: name for converted model files
        output_directory: directory for writing result
        model: path to the topology file
        weights: path to the weights file
        framework: framework name for original model
        mo_search_paths: paths where ModelOptimizer may be found. If None only default paths is used
        mo_params: value parameters for ModelOptimizer execution
        mo_flags: flags parameters for ModelOptimizer execution
    Returns:
        paths to converted to IE IR model and weights
    """
    set_topology_name(mo_params, topology_name)
    set_output_directory(mo_params, output_directory)
    model_optimizer_executable = find_mo(mo_search_paths)
    if model_optimizer_executable is None:
        raise EnvironmentError(
            "Model optimizer not found. Please set MO_DIR environment variable to model optimizer folder "
            "installation or refer to help for command line options for providing Model optimizer"
        )

    set_path_to_custom_operation_configs(mo_params, framework, tf_config_dir, model_optimizer_executable)
    mo_flags = list(map(format_key, mo_flags))
    mo_params = prepare_mo_params(mo_params)

    args = {'--framework': framework}
    framework_specific_options = {
        'caffe': {'--input_model': weights, '--input_proto': model},
        'mxnet': {'--input_model': weights},
        'tf': {'--input_model': model},
        'onnx': {'--input_model': model},
        'kaldi': {'--input_model': model}
    }
    args.update(framework_specific_options.get(framework, {}))

    args.update(mo_params)
    args = prepare_args(str(model_optimizer_executable), flag_options=mo_flags, value_options=args)
    code = exec_mo_binary(args)

    if code.returncode != 0:
        raise RuntimeError("Model optimizer conversion failed: ModelOptimizer returned non-zero code")

    model_file, bin_file = find_dlsdk_ir(mo_params['--output_dir'], mo_params['--model_name'])

    if bin_file is None or model_file is None:
        raise RuntimeError("Model optimizer finished correctly, but converted model is not found.")

    return model_file, bin_file


def find_dlsdk_ir(search_path: Path, model_name):
    """
    Args:
        search_path: path with IE IR of model
        model_name: name of the model
    Returns:
        paths to IE IR of model
    """
    xml_file = search_path / '{}.xml'.format(model_name)
    bin_file = search_path / '{}.bin'.format(model_name)

    return get_path(xml_file), get_path(bin_file)


def find_mo(search_paths=None) -> Union[Path, None]:
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

        return mo

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
    for k, v in mo_params.items():
        updated_mo_params[format_key(k)] = v
    return updated_mo_params


def set_path_to_custom_operation_configs(mo_params, framework, config_directory, mo_path):
    if framework != 'tf':
        return mo_params

    config_path = mo_params.get('tensorflow_use_custom_operations_config')
    if not config_path:
        return mo_params

    if config_directory is not None:
        config_directory = Path(config_directory)
    else:
        config_directory = Path('/').joinpath(*mo_path.parts[:-1]) / 'extensions' / 'front' / 'tf'

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = config_directory / config_path

    mo_params['tensorflow_use_custom_operations_config'] = get_path(config_path)
    return mo_params


def set_topology_name(mo_params, topology_name):
    if mo_params.get('model_name') is None:
        mo_params['model_name'] = topology_name
    return mo_params


def set_output_directory(mo_params, output_directory):
    mo_params['output_dir'] = output_directory
    return mo_params
