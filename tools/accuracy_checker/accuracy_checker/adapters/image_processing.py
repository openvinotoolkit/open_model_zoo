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
from PIL import Image
import cv2
import numpy as np

from ..adapters import Adapter
from ..representation import ImageProcessingPrediction, SuperResolutionPrediction, ContainerPrediction
from ..config import ConfigValidator, BoolField, BaseField, StringField, DictField, ConfigError
from ..utils import get_or_parse_value
from ..preprocessor import Normalize


class ImageProcessingAdapter(Adapter):
    __provider__ = 'image_processing'

    prediction_types = (ImageProcessingPrediction, )

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

        for identifier, out_img in zip(identifiers, raw_outputs[self.target_out]):
            out_img = self._basic_postprocess(out_img)
            result.append(SuperResolutionPrediction(identifier, out_img))

        return result

    def _basic_postprocess(self, img):
        img *= self.std
        img += self.mean
        img = img.transpose((1, 2, 0)) if img.shape[-1] not in [3, 4, 1] else img
        if self.cast_to_uint8:
            img = np.clip(img, 0., 255.)
            img = img.astype(np.uint8)
        if self.reverse_channels:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img, 'RGB') if Image is not None else img
            img = np.array(img).astype(np.uint8)

        return img


class SuperResolutionAdapter(ImageProcessingAdapter):
    __provider__ = 'super_resolution'
    prediction_types = (SuperResolutionPrediction, )

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        raw_outputs = self._extract_predictions(raw, frame_meta)
        if not self.target_out:
            self.target_out = self.output_blob

        for identifier, img_sr in zip(identifiers, raw_outputs[self.target_out]):
            img_sr = self._basic_postprocess(img_sr)
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
            "cast_to_uint8": BoolField(
                optional=True, default=True, description="Cast prediction values to integer within [0, 255] range"
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


class SuperResolutionYUV(Adapter):
    __provider__ = 'super_resolution_yuv'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'y_output': StringField(),
            'u_output': StringField(),
            'v_output': StringField(),
            'target_color': StringField(optional=True, choices=['bgr', 'rgb'], default='bgr')
        })
        return parameters

    def configure(self):
        self.y_output = self.get_value_from_config('y_output')
        self.u_output = self.get_value_from_config('u_output')
        self.v_output = self.get_value_from_config('v_output')
        self.color = cv2.COLOR_YUV2BGR if self.get_value_from_config('target_color') == 'bgr' else cv2.COLOR_YUV2RGB

    def get_image(self, y, u, v):
        is_hwc = u.shape[-1] == 1
        if not is_hwc:
            y = np.transpose(y, (1, 2, 0))
            u = np.transpose(u, (1, 2, 0))
            v = np.transpose(v, (1, 2, 0))
        h, w, __ = u.shape
        u = u.reshape(h, w, 1)
        v = v.reshape(h, w, 1)
        u = cv2.resize(u, None, fx=2, fy=2)
        v = cv2.resize(v, None, fx=2, fy=2)

        y = y.reshape(2 * h, 2 * w, 1)
        u = u.reshape(2 * h, 2 * w, 1)
        v = v.reshape(2 * h, 2 * w, 1)
        yuv = np.concatenate([y, u, v], axis=2)
        image = cv2.cvtColor(yuv, self.color)
        return image

    def process(self, raw, identifiers=None, frame_meta=None):
        outs = self._extract_predictions(raw, frame_meta)
        results = []
        for identifier, yres, ures, vres in zip(
                identifiers, outs[self.y_output], outs[self.u_output], outs[self.v_output]
        ):
            sr_img = self.get_image(yres, ures, vres)
            results.append(SuperResolutionPrediction(identifier, sr_img))

        return results
