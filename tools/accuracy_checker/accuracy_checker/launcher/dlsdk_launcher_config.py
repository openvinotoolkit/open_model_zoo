"""
<<<<<<< 2d7f7598dcd544df76f7aae81406d9ffebf235df
Copyright (c) 2019-2020 Intel Corporation
=======
Copyright (c) 2020 Intel Corporation
>>>>>>> refactor launcher, add condition for switching to sync mode

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
from ..config import PathField, ConfigError
from .launcher import LauncherConfigValidator
from .model_conversion import FrameworkParameters
from ..logging import warning
from ..utils import get_path, contains_all

try:
    from openvino.inference_engine import known_plugins
except ImportError:
    known_plugins = []

HETERO_KEYWORD = 'HETERO:'
MULTI_DEVICE_KEYWORD = 'MULTI:'
FPGA_COMPILER_MODE_VAR = 'CL_CONTEXT_COMPILER_MODE_INTELFPGA'
NIREQ_REGEX = r"(\(\d+\))"
VPU_PLUGINS = ('HDDL', "MYRIAD")
VPU_LOG_LEVELS = ('LOG_NONE', 'LOG_WARNING', 'LOG_INFO', 'LOG_DEBUG')


class CPUExtensionPathField(PathField):
    def __init__(self, **kwargs):
        super().__init__(is_directory=False, **kwargs)

    def validate(self, entry, field_uri=None):
        if entry is None:
            return

        field_uri = field_uri or self.field_uri
        validation_entry = ''
        try:
            validation_entry = Path(entry)
        except TypeError:
            self.raise_error(entry, field_uri, "values is expected to be path-like")
        is_directory = False
        if validation_entry.parts[-1] == 'AUTO':
            validation_entry = validation_entry.parent
            is_directory = True
        try:
            get_path(validation_entry, is_directory)
        except FileNotFoundError:
            self.raise_error(validation_entry, field_uri, "path does not exist")
        except NotADirectoryError:
            self.raise_error(validation_entry, field_uri, "path is not a directory")
        except IsADirectoryError:
            self.raise_error(validation_entry, field_uri, "path is a directory, regular file expected")


class DLSDKLauncherConfigValidator(LauncherConfigValidator):
    def __init__(
            self, config_uri, fields=None, delayed_model_loading=False, **kwargs
    ):
        super().__init__(config_uri, fields, delayed_model_loading, **kwargs)
        self.need_conversion = None

    def create_device_regex(self, available_devices):
        self.regular_device_regex = r"(?:^(?P<device>{devices})$)".format(devices="|".join(available_devices))
        self.hetero_regex = r"(?:^{hetero}(?P<devices>(?:{devices})(?:,(?:{devices}))*)$)".format(
            hetero=HETERO_KEYWORD, devices="|".join(available_devices)
        )
        self.multi_device_regex = r"(?:^{multi}(?P<devices_ireq>(?:{devices_ireq})(?:,(?:{devices_ireq}))*)$)".format(
            multi=MULTI_DEVICE_KEYWORD, devices_ireq="{}?|".format(NIREQ_REGEX).join(available_devices)
        )
        self.supported_device_regex = r"{multi}|{hetero}|{regular}".format(
            multi=self.multi_device_regex, hetero=self.hetero_regex, regular=self.regular_device_regex
        )
        self.fields['device'].set_regex(self.supported_device_regex)

    def validate(self, entry, field_uri=None, ie_core=None):
        """
        Validate that launcher entry meets all configuration structure requirements.
        Args:
            entry: launcher configuration file entry.
            field_uri: id of launcher entry.
            ie_core: IECore instance.
        """
        if not self.delayed_model_loading:
            framework_parameters = self.check_model_source(entry)
            self._set_model_source(framework_parameters)
        super().validate(entry, field_uri)
        self.create_device_regex(known_plugins)
        try:
            self.fields['device'].validate(entry['device'], field_uri)
        except ConfigError as error:
            if ie_core is not None:
                self.create_device_regex(ie_core.available_devices)
                try:
                    self.fields['device'].validate(entry['device'], field_uri)
                except ConfigError:
                    # workaround for devices where this metric is non implemented
                    warning('unknown device: {}'.format(entry['device']))
            else:
                raise error

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
    def check_model_source(entry):
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
        for mo_source_option in sources:
            if contains_all(entry, sources[mo_source_option]):
                specified.append(mo_source_option)

        if not specified:
            raise ConfigError('{} None provided'.format(multiple_model_sources_err))
        if len(specified) > 1:
            raise ConfigError('{} Several provided'.format(multiple_model_sources_err))

        return specified[0]
