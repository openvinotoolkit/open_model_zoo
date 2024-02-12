"""
Copyright (c) 2018-2024 Intel Corporation

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

from argparse import ArgumentParser
from functools import partial
from pathlib import Path

from . import __version__
from .utils import get_path, cast_to_bool, ov_new_api_available


def add_common_args(parser):
    common_args = parser.add_argument_group('Common arguments')
    common_args.add_argument(
        '-d', '--definitions',
        help='path to the yml file with definitions',
        type=get_path,
        required=False
    )
    common_args.add_argument(
        '-c', '--config',
        help='path to the yml file with local configuration',
        type=get_path,
        required=True
    )
    common_args.add_argument(
        '-m', '--models',
        help='prefix path to the models and weights',
        type=partial(get_path, file_or_directory=True),
        required=False,
        nargs='+'
    )
    common_args.add_argument(
        '-s', '--source',
        help='prefix path to the data source',
        type=partial(get_path, is_directory=True),
        required=False
    )
    common_args.add_argument(
        '-a', '--annotations',
        help='prefix path to the converted annotations and datasets meta data',
        type=partial(get_path, is_directory=True),
        required=False
    )
    common_args.add_argument(
        '--model_attributes',
        help="path's prefix for additional models attributes",
        type=partial(get_path, is_directory=True),
        required=False
    )
    common_args.add_argument(
        '--input_precision',
        help='space-separated list of precisions for network inputs. '
             'Providing several values required <layer_name>:<precision> format. '
             'If single value without layer_name provided, then it will be applied to all input layers.',
        required=False,
        nargs='+'
    )
    common_args.add_argument(
        '--layout',
        help='Prompts how network layouts should be treated by application.'
             'For example, "input1[NCHW],input2[NC]" or "[NCHW]" in case of one input size.',
        required=False
    )


def add_config_filtration_args(parser):
    config_filtration_args = parser.add_argument_group('Config filtration arguments')
    config_filtration_args.add_argument(
        '-tf', '--target_framework',
        help='framework for infer',
        required=False
    )
    config_filtration_args.add_argument(
        '-td', '--target_devices',
        help='space separated list of devices for infer',
        required=False,
        nargs='+'
    )
    config_filtration_args.add_argument(
        '-tt', '--target_tags',
        help='space separated list of launcher tags for infer',
        required=False,
        nargs='+'
    )
    config_filtration_args.add_argument(
        '-tb', '--target_backends', help='space separated list of backends for inference',
        required=False,
        nargs='+'
    )


def add_dataset_related_args(parser):
    dataset_related_args = parser.add_argument_group('Dataset related arguments')
    dataset_related_args.add_argument(
        '-ss', '--subsample_size',
        help="dataset subsample size",
        type=str,
        required=False
    )
    dataset_related_args.add_argument(
        '--shuffle',
        help="allow shuffle annotation during creation a subset",
        type=cast_to_bool,
        required=False
    )
    dataset_related_args.add_argument(
        '--store_subset',
        help='allow to save evaluation data ids',
        type=cast_to_bool,
        default=False,
        required=False
    )
    dataset_related_args.add_argument(
        '--subset_file',
        help='file name for saving or reading identifiers subset',
        required=False
    )
    dataset_related_args.add_argument(
        '--sub_evaluation',
        help='attempt to use subset size and metrics for sub evaluation',
        type=cast_to_bool,
        default=False,
        required=False
    )


def add_profiling_related_args(parser):
    profiling_related_args = parser.add_argument_group('Profiling related arguments')
    profiling_related_args.add_argument(
        '--profile',
        help='activate metric profiling mode',
        type=cast_to_bool,
        required=False
    )
    profiling_related_args.add_argument(
        '--profiler_logs_dir',
        help='path to save profiler logs',
        type=partial(get_path, is_directory=True),
        default=Path.cwd(),
        required=False
    )
    profiling_related_args.add_argument(
        '--profile_report_type',
        help='report type for profiler logs',
        default='csv',
        choices=['csv', 'json'],
        required=False
    )


def add_tool_settings_args(parser):
    tool_settings_args = parser.add_argument_group('Tool settings arguments')
    tool_settings_args.add_argument(
        '--progress',
        help='progress reporter. You can select bar or print',
        default='bar',
        required=False
    )
    tool_settings_args.add_argument(
        '--progress_interval',
        help='interval for update progress if selected *print* progress.',
        type=int,
        default=1000,
        required=False
    )
    tool_settings_args.add_argument(
        '--ignore_result_formatting',
        help='allow to get raw metrics results without data formatting',
        type=cast_to_bool,
        default=False,
        required=False
    )
    tool_settings_args.add_argument(
        '--ignore_metric_reference', help='disable comparing with metric reference during presenting result',
        type=cast_to_bool,
        default=False,
        required=False
    )
    tool_settings_args.add_argument(
        '--stored_predictions',
        help='path to file with saved predictions. Used for development',
        # since at the first time file does not exist and then created we can not always check existence
        required=False
    )
    tool_settings_args.add_argument(
        '--csv_result',
        help='file for results writing',
        required=False,
    )
    tool_settings_args.add_argument(
        '--intermediate_metrics_results',
        help='enables intermediate metrics results printing or saving',
        type=cast_to_bool,
        default=False,
        required=False
    )
    tool_settings_args.add_argument(
        '--metrics_interval',
        help='number of iteration for updated metrics result printing',
        type=int,
        default=1000,
        required=False
    )
    tool_settings_args.add_argument(
        '--store_only',
        type=cast_to_bool,
        default=False,
        required=False
    )
    tool_settings_args.add_argument(
        '-l', '--log_file',
        help='file for additional logging results',
        required=False
    )


def add_openvino_specific_args(parser):
    openvino_specific_args = parser.add_argument_group('OpenVINO specific arguments')
    openvino_specific_args.add_argument(
        '-e', '--extensions',
        help='prefix path to extensions folder',
        type=partial(get_path, check_exists=False),
        default=Path.cwd(),
        required=False
    )
    openvino_specific_args.add_argument(
        '--cpu_extensions_mode',
        help='specified preferable set of processor instruction for automatic searching cpu extension lib',
        choices=['avx512', 'avx2', 'sse4'],
        required=False
    )
    openvino_specific_args.add_argument(
        '-dc', '--device_config',
        help='Inference Engine device specific config file',
        type=get_path,
        required=False
    )
    openvino_specific_args.add_argument(
        '--ie_preprocessing',
        help='enable preprocessing via Inference Engine. Accepted only for dlsdk launcher.',
        type=cast_to_bool,
        default=False,
        required=False
    )
    openvino_specific_args.add_argument(
        '--model_is_blob',
        help='the tip for automatic model search to use blob for dlsdk launcher',
        type=cast_to_bool,
        required=False
    )
    openvino_specific_args.add_argument(
        '--model_type',
        help='model format for automatic search (e.g. blob, xml, onnx)',
        required=False
    )
    openvino_specific_args.add_argument(
        '-am', '--affinity_map',
        help='prefix path to the affinity maps',
        type=partial(get_path, file_or_directory=True),
        default=Path.cwd(),
        required=False
    )
    openvino_specific_args.add_argument(
        '--vpu_log_level',
        help='log level for VPU devices',
        default='LOG_WARNING',
        choices=['LOG_NONE', 'LOG_WARNING', 'LOG_INFO', 'LOG_DEBUG'],
        required=False
    )
    openvino_specific_args.add_argument(
        '--async_mode',
        help='Allow evaluation in async mode',
        type=cast_to_bool,
        default=False,
        required=False
    )
    openvino_specific_args.add_argument(
        '--num_requests',
        help='the number of infer requests',
        required=False
    )
    openvino_specific_args.add_argument(
        '--kaldi_bin_dir', help='directory with Kaldi utility binaries. Required only for Kaldi models decoding.',
        required=False, type=partial(get_path, is_directory=True)
    )
    openvino_specific_args.add_argument(
        '--kaldi_log_file', help='path for saving logs from Kaldi tools', type=partial(get_path, check_exists=False),
        required=False
    )
    openvino_specific_args.add_argument(
        '--undefined_shapes_resolving_policy', choices=['default', 'dynamic', 'static'],
        help='Policy how to make deal with undefined shapes in network: '
             'default - try to run as default, if does not work switch to static, '
             'dynamic - enforce network execution with dynamic shapes, '
             'static - convert undefined shapes to static before execution',
        required=False, default='default'
    )
    openvino_specific_args.add_argument(
        '--inference_precision_hint',
        help='Inference Precision hint for device',
        required=False
    )
    openvino_specific_args.add_argument(
        '--use_new_api', type=cast_to_bool, help='switch to processing using OpenVINO 2.0 API', required=False,
        default=ov_new_api_available()
    )


def build_arguments_parser():
    parser = ArgumentParser(description='Deep Learning accuracy validation framework', allow_abbrev=False)
    add_common_args(parser)
    add_config_filtration_args(parser)
    add_dataset_related_args(parser)
    add_profiling_related_args(parser)
    add_tool_settings_args(parser)
    add_openvino_specific_args(parser)

    parser.add_argument(
        '--version',
        help='show tool version and exit',
        action='version',
        version='%(prog)s {version}'.format(version=__version__)
    )

    return parser
