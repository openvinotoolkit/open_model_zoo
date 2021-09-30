"""
Copyright (c) 2018-2021 Intel Corporation

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
from pathlib import Path
import platform
from ..config import PathField, ConfigError, StringField, NumberField, ListField, DictField, BaseField, BoolField
from .launcher import LauncherConfigValidator
from .model_conversion import FrameworkParameters, convert_model
from ..logging import warning, print_info
from ..utils import get_path, contains_all, UnsupportedPackage, get_parameter_value_from_config, string_to_tuple

try:
    from openvino.inference_engine import known_plugins  # pylint:disable=W9902
except ImportError:
    known_plugins = []

try:
    from cpuinfo import get_cpu_info
except ImportError as import_error:
    get_cpu_info = UnsupportedPackage("cpuinfo", import_error.msg)

HETERO_KEYWORD = 'HETERO:'
MULTI_DEVICE_KEYWORD = 'MULTI:'
FPGA_COMPILER_MODE_VAR = 'CL_CONTEXT_COMPILER_MODE_INTELFPGA'
NIREQ_REGEX = r"(\(\d+\))"
VPU_PLUGINS = ('HDDL', "MYRIAD")
VPU_LOG_LEVELS = ('LOG_NONE', 'LOG_WARNING', 'LOG_INFO', 'LOG_DEBUG')


def parse_partial_shape(partial_shape):
    ps = str(partial_shape)
    preprocessed = ps.replace('{', '(').replace('}', ')').replace('?', '-1')
    if '[' not in preprocessed:
        return string_to_tuple(preprocessed, casting_type=int)
    shape_list = []
    s_pos = 0
    e_pos = len(preprocessed)
    while s_pos >= e_pos:
        open_brace = preprocessed.find('[', s_pos, e_pos)
        if open_brace == -1:
            shape_list.extend(string_to_tuple(preprocessed[s_pos:], casting_type=int))
            break
        if open_brace != s_pos:
            shape_list.extend(string_to_tuple(preprocessed[:open_brace], casting_type=int))
        close_brace = preprocessed.find(']', open_brace, e_pos)
        shape_range = preprocessed[open_brace + 1:close_brace]
        shape_list.append(string_to_tuple(shape_range, casting_type=int))
        s_pos = min(close_brace + 2, e_pos)
    return shape_list


class CPUExtensionPathField(PathField):
    def __init__(self, **kwargs):
        super().__init__(is_directory=False, **kwargs)

    def validate(self, entry, field_uri=None, fetch_only=False, validation_scheme=None):
        errors = []
        if entry is None:
            return errors

        field_uri = field_uri or self.field_uri
        validation_entry = ''
        try:
            validation_entry = Path(entry)
        except TypeError:
            msg = "values is expected to be path-like"
            if not fetch_only:
                self.raise_error(entry, field_uri, msg)
            errors.append(self.build_error(entry, field_uri, msg, validation_scheme=validation_scheme))
        is_directory = False
        if validation_entry.parts[-1] == 'AUTO':
            validation_entry = validation_entry.parent
            is_directory = True
        try:
            get_path(validation_entry, is_directory)
        except FileNotFoundError:
            msg = "path does not exist"
            if not fetch_only:
                self.raise_error(validation_entry, field_uri, msg)
            errors.append(self.build_error(validation_entry, field_uri, msg, validation_scheme=validation_scheme))
        except NotADirectoryError:
            msg = "path is not a directory"
            if not fetch_only:
                self.raise_error(validation_entry, field_uri, msg)
            errors.append(self.build_error(validation_entry, field_uri, msg, validation_scheme=validation_scheme))
        except IsADirectoryError:
            msg = "path is a directory, regular file expected"
            if not fetch_only:
                self.raise_error(validation_entry, field_uri, msg)
            errors.append(self.build_error(validation_entry, field_uri, msg, validation_scheme=validation_scheme))
        return errors


class DLSDKLauncherConfigValidator(LauncherConfigValidator):
    def __init__(self, config_uri, fields=None, delayed_model_loading=False, **kwargs):
        super().__init__(config_uri, fields, delayed_model_loading, **kwargs)
        self.need_conversion = None

    def create_device_regex(self, available_devices):
        resolve_multi_device = set()
        for device in available_devices:
            resolve_multi_device.add(device)
            if '.' in device:
                resolve_multi_device.add(device.split('.')[0])
        self.regular_device_regex = r"(?:^(?P<device>{devices})$)".format(devices="|".join(resolve_multi_device))
        self.hetero_regex = r"(?:^{hetero}(?P<devices>(?:{devices})(?:,(?:{devices}))*)$)".format(
            hetero=HETERO_KEYWORD, devices="|".join(resolve_multi_device)
        )
        self.multi_device_regex = r"(?:^{multi}(?P<devices_ireq>(?:{devices_ireq})(?:,(?:{devices_ireq}))*)$)".format(
            multi=MULTI_DEVICE_KEYWORD, devices_ireq="{}?|".format(NIREQ_REGEX).join(resolve_multi_device)
        )
        self.supported_device_regex = r"{multi}|{hetero}|{regular}".format(
            multi=self.multi_device_regex, hetero=self.hetero_regex, regular=self.regular_device_regex
        )
        self.fields['device'].set_regex(self.supported_device_regex)

    def validate(self, entry, field_uri=None, ie_core=None, fetch_only=False, validation_scheme=None):
        """
        Validate that launcher entry meets all configuration structure requirements.
        Args:
            entry: launcher configuration file entry.
            field_uri: id of launcher entry.
            ie_core: IECore instance.
            fetch_only: only fetch possible error without raising
            validation_scheme: scheme for validation
        """
        error_stack = []
        if not self.delayed_model_loading:
            framework_parameters, error_stack = self.check_model_source(entry, fetch_only, field_uri, validation_scheme)
            if not error_stack:
                self._set_model_source(framework_parameters)
        error_stack += super().validate(entry, field_uri, fetch_only, validation_scheme)
        self.create_device_regex(known_plugins)
        if 'device' not in entry:
            return error_stack
        try:
            self.fields['device'].validate(
                entry['device'], field_uri, validation_scheme=(validation_scheme or {}).get('device')
            )
        except ConfigError as error:
            if ie_core is not None:
                self.create_device_regex(ie_core.available_devices)
                try:
                    self.fields['device'].validate(
                        entry['device'], field_uri, validation_scheme=(validation_scheme or {}).get('device')
                    )
                except ConfigError:
                    # workaround for devices where this metric is non implemented
                    warning('unknown device: {}'.format(entry['device']))
            else:
                if not fetch_only:
                    raise error
                error_stack.append(error)
        return error_stack

    def _set_model_source(self, framework):
        self.need_conversion = framework.name != 'dlsdk'
        self.framework = framework
        self.fields['model'].optional = self.need_conversion
        self.fields['caffe_model'].optional = framework.name != 'caffe'
        self.fields['caffe_weights'].optional = framework.name != 'caffe'
        self.fields['mxnet_weights'].optional = framework.name != 'mxnet'
        self.fields['tf_model'].optional = framework != FrameworkParameters('tf', False)
        self.fields['tf_meta'].optional = framework != FrameworkParameters('tf', True)
        self.fields['onnx_model'].optional = framework.name != 'onnx'
        self.fields['kaldi_model'].optional = framework.name != 'kaldi'

    @staticmethod
    def check_model_source(entry, fetch_only=False, field_uri=None, validation_scheme=None):
        dlsdk_model_options = ['model']
        caffe_model_options = ['caffe_model', 'caffe_weights']
        mxnet_model_options = ['mxnet_weights']
        tf_model_options = ['tf_model']
        tf_meta_options = ['tf_meta']
        onnx_model_options = ['onnx_model']
        kaldi_model_options = ['kaldi_model']

        multiple_model_sources_err = (
            'Either model and weights or caffe_model and caffe_weights '
            'or mxnet_weights or tf_model or tf_meta should be specified.'
        )
        sources = {
            FrameworkParameters('dlsdk', False): dlsdk_model_options,
            FrameworkParameters('caffe', False): caffe_model_options,
            FrameworkParameters('tf', False): tf_model_options,
            FrameworkParameters('mxnet', False): mxnet_model_options,
            FrameworkParameters('onnx', False): onnx_model_options,
            FrameworkParameters('kaldi', False): kaldi_model_options,
            FrameworkParameters('tf', True): tf_meta_options
        }

        specified = []
        for mo_source_option, mo_source_value in sources.items():
            if contains_all(entry, mo_source_value):
                specified.append(mo_source_option)

        if not specified:
            error = ConfigError(
                '{} None provided'.format(multiple_model_sources_err), entry, field_uri, validation_scheme
            )
            if not fetch_only:
                raise error
            return None, [error]
        if len(specified) > 1:
            error = ConfigError(
                '{} Several provided'.format(multiple_model_sources_err), entry, field_uri, validation_scheme
            )
            if not fetch_only:
                raise error
            return None, [error]

        return specified[0], []


DLSDK_LAUNCHER_PARAMETERS = {
    'model': PathField(description="Path to model.", file_or_directory=True),
    'weights': PathField(description="Path to weights.", optional=True, file_or_directory=True),
    'device': StringField(description="Device name."),
    'caffe_model': PathField(optional=True, description="Path to Caffe model file."),
    'caffe_weights': PathField(optional=True, description="Path to Caffe weights file."),
    'mxnet_weights': PathField(optional=True, description="Path to MXNet weights file."),
    'tf_model': PathField(optional=True, description="Path to TF model file."),
    'tf_meta': PathField(optional=True, description="Path to TF meta file."),
    'onnx_model': PathField(optional=True, description="Path to ONNX model file."),
    'kaldi_model': PathField(optional=True, description="Path to Kaldi model file."),
    'cpu_extensions': CPUExtensionPathField(optional=True, description="Path to CPU extensions."),
    'gpu_extensions': PathField(optional=True, description="Path to GPU extensions."),
    'bitstream': PathField(optional=True, description="Bitream (FPGA only)."),
    'mo_params': DictField(optional=True, description="Model Optimizer parameters."),
    'mo_flags': ListField(optional=True, description="Model Optimizer flags."),
    'outputs': ListField(optional=True, description="Outputs."),
    'allow_reshape_input': BoolField(optional=True, default=False, description="Allows reshape input."),
    'affinity_map': PathField(optional=True, description="Affinity map."),
    'batch': NumberField(value_type=int, min_value=1, optional=True, default=1, description="Batch size."),
    'should_log_cmd': BoolField(optional=True, description="Log Model Optimizer command."),
    'async_mode': BoolField(optional=True, description="Allows asynchronous mode.", default=False),
    'num_requests': BaseField(
        optional=True,
        description="Number of requests (for async mode only). "
                    "In multi device mode allows setting comma-separated list for numbers "
                    "or one value which will be used for all devices"
    ),
    'reset_memory_state': BoolField(
        optional=True, default=False,
        description='Reset infer request memory states after inference. '
                    'State control essential for recurrent networks'),
    'device_config': DictField(optional=True, description='device configuration'),
    '_model_optimizer': PathField(optional=True, is_directory=True, description="Model optimizer."),
    '_tf_obj_detection_api_config_dir': PathField(
        optional=True, is_directory=True, description="TF Object Detection API Config."
    ),
    '_tf_custom_op_config_dir': PathField(
        optional=True, is_directory=True, description="TF Custom Operation Config prefix."
    ),
    '_transformations_config_dir': PathField(
        optional=True, is_directory=True, description="Transformation config prefix for Model Optimizer"),
    '_tf_obj_detection_api_pipeline_config_path': PathField(
        optional=True, is_directory=False, description="TF Custom Operation Pipeline Config."),
    '_cpu_extensions_mode': StringField(optional=True, description="CPU extensions mode."),
    '_aocl': PathField(optional=True, description="path to aocl (FPGA only)"),
    '_vpu_log_level': StringField(
        optional=True, choices=VPU_LOG_LEVELS, description="VPU LOG level: {}".format(', '.join(VPU_LOG_LEVELS))
    ),
    '_prev_bitstream': PathField(optional=True, description="path to bitstream from previous run (FPGA only)"),
    '_model_is_blob': BoolField(optional=True, description='hint for auto model search'),
    '_undefined_shapes_resolving_policy': StringField(
        optional=True, default='default', choices=['default', 'dynamic', 'static'],
        description='Policy how to make deal with undefined shapes in network: '
                    'default - try to run as default, if does not work switch to static, '
                    'dynamic - enforce network execution with dynamic shapes, '
                    'static - convert undefined shapes to static before execution'
    )
}


def get_cpu_extension(cpu_extensions, selection_mode):
    def get_cpu_extensions_list(file_format, base_name, selection_mode):
        if not selection_mode:
            default_cpu_extension = file_format.format(base_name)
            extension_list = list(extensions_path.glob(default_cpu_extension))

            if extension_list:
                return extension_list

            if isinstance(get_cpu_info, UnsupportedPackage):
                get_cpu_info.raise_error("CPU extensions automatic search")

            cpu_info_flags = get_cpu_info()['flags']
            supported_flags = ['avx512', 'avx2', 'sse4_1', 'sse4_2']
            cpu_info_flag_to_suffix = {
                'avx512': 'avx512',
                'avx2': 'avx2',
                'sse4_1': 'sse4',
                'sse4_2': 'sse4'
            }
            for flag in supported_flags:
                selection_mode = cpu_info_flag_to_suffix[flag]
                if flag in cpu_info_flags:
                    break

        extension_list = list(extensions_path.glob(file_format.format('{}_{}'.format(base_name, selection_mode))))
        return extension_list

    os_specific_formats = {
        'Darwin': ('lib{}.dylib', 'lib{}.so'),
        'Linux': ('lib{}.so',),
        'Windows': ('{}.dll',),
    }
    cpu_extensions_name = cpu_extensions.parts[-1]
    if cpu_extensions_name != 'AUTO':
        return cpu_extensions
    extensions_path = cpu_extensions.parent
    system_name = platform.system()
    file_formats = os_specific_formats.get(system_name)
    if not file_formats:
        raise ConfigError(
            'Accuracy Checker can not automatically find cpu extensions library '
            'for {} platform. Please, set cpu extension library manually.'.format(system_name)
        )
    extension_list = []
    for supported_format in file_formats:
        extension_list = get_cpu_extensions_list(supported_format, 'cpu_extension', selection_mode)
        if extension_list:
            break
    if not extension_list:
        raise ConfigError('suitable CPU extension lib not found in {}'.format(extensions_path))
    return extension_list[0]


def mo_convert_model(config, launcher_parameters, framework=None):
    if framework is None:
        framework = DLSDKLauncherConfigValidator.check_model_source(config)
    config_model = config.get('{}_model'.format(framework.name), '')
    config_weights = config.get('{}_weights'.format(framework.name), '')
    config_meta = config.get('{}_meta'.format(framework.name), '')
    mo_search_paths = []
    model_optimizer = get_parameter_value_from_config(config, launcher_parameters, '_model_optimizer')
    if model_optimizer:
        mo_search_paths.append(model_optimizer)
    model_optimizer_directory_env = os.environ.get('MO_DIR')
    if model_optimizer_directory_env:
        mo_search_paths.append(model_optimizer_directory_env)
    model_name = (
        Path(config_model).name.rsplit('.', 1)[0] or
        Path(config_weights).name.rsplit('.', 1)[0] or
        Path(config_meta).name.rsplit('.', 1)[0]
    )
    should_log_mo_cmd = get_parameter_value_from_config(config, launcher_parameters, 'should_log_cmd')

    return convert_model(
        model_name,
        config_model, config_weights, config_meta, framework, mo_search_paths,
        get_parameter_value_from_config(config, launcher_parameters, 'mo_params'),
        get_parameter_value_from_config(config, launcher_parameters, 'mo_flags'),
        get_parameter_value_from_config(config, launcher_parameters, '_tf_custom_op_config_dir'),
        get_parameter_value_from_config(
            config, launcher_parameters, '_tf_obj_detection_api_pipeline_config_path'
        ),
        get_parameter_value_from_config(config, launcher_parameters, '_transformations_config_dir'),
        should_log_cmd=should_log_mo_cmd
    )


def automatic_model_search(model_name, model_cfg, weights_cfg, model_is_blob):
    def get_xml(model_dir):
        models_list = list(model_dir.glob('{}.xml'.format(model_name)))
        if not models_list:
            models_list = list(model_dir.glob('*.xml'))
        return models_list

    def get_blob(model_dir):
        blobs_list = list(Path(model_dir).glob('{}.blob'.format(model_name)))
        if not blobs_list:
            blobs_list = list(Path(model_dir).glob('*.blob'))
        return blobs_list

    def get_onnx(model_dir):
        onnx_list = list(Path(model_dir).glob('{}.onnx'.format(model_name)))
        if not onnx_list:
            onnx_list = list(Path(model_dir).glob('*.onnx'))
        return onnx_list

    def get_model():
        model = Path(model_cfg)
        if not model.is_dir():
            accepted_suffixes = ['.blob', '.onnx', '.xml']
            if model.suffix not in accepted_suffixes:
                raise ConfigError('Models with following suffixes are allowed: {}'.format(accepted_suffixes))
            print_info('Found model {}'.format(model))
            return model, model.suffix == '.blob'
        if model_is_blob:
            model_list = get_blob(model)
        else:
            model_list = get_xml(model)
            if not model_list and model_is_blob is None:
                model_list = get_blob(model)
            if not model_list:
                model_list = get_onnx(model)
        if not model_list:
            raise ConfigError('suitable model is not found')
        if len(model_list) != 1:
            raise ConfigError('More than one model matched, please specify explicitly')
        model = model_list[0]
        print_info('Found model {}'.format(model))
        return model, model.suffix == '.blob'

    model, is_blob = get_model()
    if is_blob:
        return model, None
    weights = weights_cfg
    if (weights is None or Path(weights).is_dir()) and model.suffix != '.onnx':
        weights_dir = weights or model.parent
        weights = Path(weights_dir) / model.name.replace('xml', 'bin')
    if weights is not None:
        accepted_weights_suffixes = ['.bin']
        if weights.suffix not in accepted_weights_suffixes:
            raise ConfigError('Weights with following suffixes are allowed: {}'.format(accepted_weights_suffixes))
        print_info('Found weights {}'.format(get_path(weights)))

    return model, weights
