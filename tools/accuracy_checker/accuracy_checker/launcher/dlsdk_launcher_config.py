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

from pathlib import Path
import platform
from ..config import PathField, ConfigError, StringField, NumberField, ListField, DictField, BaseField, BoolField
from .launcher import LauncherConfigValidator
from ..logging import warning, print_info
from ..utils import get_path, contains_all, UnsupportedPackage

try:
    from openvino.inference_engine import known_plugins  # pylint: disable=import-outside-toplevel,package-absolute-imports
except ImportError:
    known_plugins = []

try:
    from cpuinfo import get_cpu_info
except ImportError as import_error:
    get_cpu_info = UnsupportedPackage("cpuinfo", import_error.msg)

HETERO_KEYWORD = 'HETERO:'
MULTI_DEVICE_KEYWORD = 'MULTI:'
AUTO_SINGLE_DEVICE_KEYWORD = "AUTO"
AUTO_DEVICE_KEYWORD = "AUTO:"
NIREQ_REGEX = r"(\(\d+\))"
VPU_PLUGINS = ('HDDL', "MYRIAD")
VPU_LOG_LEVELS = ('LOG_NONE', 'LOG_WARNING', 'LOG_INFO', 'LOG_DEBUG')


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
            _, error_stack = self.check_model_source(entry, fetch_only, field_uri, validation_scheme)
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

    @staticmethod
    def check_model_source(entry, fetch_only=False, field_uri=None, validation_scheme=None):
        ov_model_options = ['model']
        sources = {'dlsdk': ov_model_options}

        specified = []
        for mo_source_option, mo_source_value in sources.items():
            if contains_all(entry, mo_source_value):
                specified.append(mo_source_option)

        if not specified:
            error = ConfigError('model is not provided', entry, field_uri, validation_scheme)
            if not fetch_only:
                raise error
            return None, [error]

        return specified[0], []

DLSDK_LAUNCHER_PARAMETERS = {
    'model': PathField(description="Path to model.", file_or_directory=True),
    'weights': PathField(description="Path to weights.", optional=True, file_or_directory=True),
    'device': StringField(description="Device name."),
    'cpu_extensions': CPUExtensionPathField(optional=True, description="Path to CPU extensions."),
    'gpu_extensions': PathField(optional=True, description="Path to GPU extensions."),
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
    '_cpu_extensions_mode': StringField(optional=True, description="CPU extensions mode."),
    '_vpu_log_level': StringField(
        optional=True, choices=VPU_LOG_LEVELS, description="VPU LOG level: {}".format(', '.join(VPU_LOG_LEVELS))
    ),
    '_model_is_blob': BoolField(optional=True, description='hint for auto model search'),
    '_undefined_shapes_resolving_policy': StringField(
        optional=True, default='default', choices=['default', 'dynamic', 'static'],
        description='Policy how to make deal with undefined shapes in network: '
                    'default - try to run as default, if does not work switch to static, '
                    'dynamic - enforce network execution with dynamic shapes, '
                    'static - convert undefined shapes to static before execution'
    ),
    '_model_type': StringField(
        choices=['xml', 'blob', 'onnx', 'paddle', 'tf', 'tflite'],
        description='hint for model type in automatic model search', optional=True),
    '_inference_precision_hint': StringField(
        description='Model execution precision for device',
        optional=True
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


def automatic_model_search(model_name, model_cfg, weights_cfg, model_type=None):
    model_type_ext = {
        'xml': 'xml',
        'blob': 'blob',
        'onnx': 'onnx',
        'paddle': 'pdmodel',
        'tf': 'pb',
        'tflite': 'tflite',
    }

    def get_model_by_suffix(model_name, model_dir, suffix):
        model_list = list(Path(model_dir).glob('{}.{}'.format(model_name, suffix)))
        if not model_list:
            model_list = list(Path(model_dir).glob('*.{}'.format(suffix)))
        return model_list

    def get_model():
        model = Path(model_cfg)
        if not model.is_dir():
            accepted_suffixes = list(model_type_ext.values())
            if model.suffix[1:] not in accepted_suffixes:
                raise ConfigError('Models with following suffixes are allowed: {}'.format(accepted_suffixes))
            print_info('Found model {}'.format(model))
            return model, model.suffix == '.blob'
        model_list = []
        if model_type is not None:
            model_list = get_model_by_suffix(model_name, model, model_type_ext[model_type])
        else:
            for ext in model_type_ext.values():
                model_list = get_model_by_suffix(model_name, model, ext)
                if model_list:
                    break
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
    if (weights is None or Path(weights).is_dir()) and model.suffix == '.xml':
        weights_dir = weights or model.parent
        weights = Path(weights_dir) / model.name.replace('xml', 'bin')
    if weights is not None:
        accepted_weights_suffixes = ['.bin']
        if weights.suffix not in accepted_weights_suffixes:
            raise ConfigError('Weights with following suffixes are allowed: {}'.format(accepted_weights_suffixes))
        print_info('Found weights {}'.format(get_path(weights)))

    return model, weights


def ov_set_config(ov_obj, config, *args, device=None, **kwargs):
    if hasattr(ov_obj, 'set_property'):
        if device is not None:
            ov_obj.set_property(device, config, *args, **kwargs)
        else:
            ov_obj.set_property(config, *args, **kwargs)
        return
    if device is not None:
        ov_obj.set_config(config, device, *args, **kwargs)
    else:
        ov_obj.set_config(config, *args, **kwargs)
