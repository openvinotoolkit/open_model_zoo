"""
Copyright (c) 2019 Intel Corporation

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
from copy import deepcopy
import cv2
import numpy as np

from ..adapters import Adapter
from ..representation import SuperResolutionPrediction, ContainerPrediction
from ..config import ConfigValidator, BoolField, BaseField, StringField, DictField, ConfigError
from ..utils import get_or_parse_value
from ..preprocessor import Normalize
try:
    from PIL import Image
except ImportError:
    Image = None


class SuperResolutionAdapter(Adapter):
    __provider__ = 'super_resolution'
    prediction_types = (SuperResolutionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'reverse_channels': BoolField(
                optional=True, default=False, description="Allow switching output image channels e.g. RGB to BGR"
            ),
            'mean': BaseField(
                optional=True, default=0,
                description='The value which should be added to prediction pixels for scaling to range [0, 255]'
                            '(usually it is the same mean value which subtracted in preprocessing step))'
            ),
            'std':  BaseField(
                optional=True, default=255,
                description='The value on which prediction pixels should be multiplied for scaling to range '
                            '[0, 255] (usually it is the same scale (std) used in preprocessing step))'
            ),
            'target_out': StringField(optional=True, description='Target super resolution model output'),
            "cast_to_uint8": BoolField(
                optional=True, default=True, description="Cast prediction values to integer within [0, 255] range"
            )
        })
        return parameters

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT)

    def configure(self):
        self.reverse_channels = self.get_value_from_config('reverse_channels')
        self.mean = get_or_parse_value(self.launcher_config.get('mean', 0), Normalize.PRECOMPUTED_MEANS)
        self.std = get_or_parse_value(self.launcher_config.get('std', 255), Normalize.PRECOMPUTED_STDS)

        if not (len(self.mean) == 3 or len(self.mean) == 1):
            raise ConfigError('mean should be one value or comma-separated list channel-wise values')

        if not (len(self.std) == 3 or len(self.std) == 1):
            raise ConfigError('std should be one value or comma-separated list channel-wise values')

        self.target_out = self.get_value_from_config('target_out')
        self.cast_to_uint8 = self.get_value_from_config('cast_to_uint8')

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        raw_outputs = self._extract_predictions(raw, frame_meta)
        if not self.target_out:
            self.target_out = self.output_blob

        for identifier, img_sr in zip(identifiers, raw_outputs[self.target_out]):
            img_sr *= self.std
            img_sr += self.mean
            if self.cast_to_uint8:
                img_sr = np.clip(img_sr, 0., 255.)
                img_sr = img_sr.astype(np.int8)
            img_sr = img_sr.transpose((1, 2, 0))
            if self.reverse_channels:
                img_sr = cv2.cvtColor(img_sr, cv2.COLOR_BGR2RGB)
                img_sr = Image.fromarray(img_sr, 'RGB') if Image is not None else img_sr
                img_sr = np.array(img_sr).astype(np.uint8)
            result.append(SuperResolutionPrediction(identifier, img_sr))

        return result


class MultiSuperResolutionAdapter(Adapter):
    __provider__ = 'multi_super_resolution'
    prediction_types = (SuperResolutionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'reverse_channels': BoolField(
                optional=True, default=False, description="Allow switching output image channels e.g. RGB to BGR"
            ),
            'mean': BaseField(
                optional=True, default=0,
                description='The value which should be added to prediction pixels for scaling to range [0, 255]'
                            '(usually it is the same mean value which subtracted in preprocessing step))'
            ),
            'std':  BaseField(
                optional=True, default=255,
                description='The value on which prediction pixels should be multiplied for scaling to range '
                            '[0, 255] (usually it is the same scale (std) used in preprocessing step))'
            ),
            'target_mapping': DictField(allow_empty=False, key_type=str, value_type=str)
        })
        return parameters

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT)

    def configure(self):
        self.target_mapping = self.get_value_from_config('target_mapping')
        common_adapter_config = deepcopy(self.launcher_config)
        self._per_target_adapters = {}
        for key, output_name in self.target_mapping.items():
            self._per_target_adapters[key] = SuperResolutionAdapter(common_adapter_config, output_blob=output_name)

    def process(self, raw, identifiers=None, frame_meta=None):
        predictions = [{}] * len(identifiers)
        for key, adapter in self._per_target_adapters.items():
            result = adapter.process(raw, identifiers, frame_meta)
            for batch_id, output_res in enumerate(result):
                predictions[batch_id][key] = output_res
        results = [ContainerPrediction(prediction_mapping) for prediction_mapping in predictions]
        return results
