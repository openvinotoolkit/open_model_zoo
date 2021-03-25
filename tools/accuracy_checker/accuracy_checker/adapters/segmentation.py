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

import math
import cv2
import numpy as np
from ..adapters import Adapter
from ..representation import SegmentationPrediction, BrainTumorSegmentationPrediction, BackgroundMattingPrediction
from ..config import ConfigError, ConfigValidator, BoolField, ListField, NumberField, StringField
from ..utils import contains_any


class SegmentationAdapter(Adapter):
    __provider__ = 'segmentation'
    prediction_types = (SegmentationPrediction,)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'make_argmax': BoolField(
                optional=True, default=False, description="Allows to apply argmax operation to output values."
            )
        })
        return parameters

    @classmethod
    def validate_config(cls, config, fetch_only=False, **kwargs):
        return super().validate_config(
            config, fetch_only=fetch_only, on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT
        )

    def configure(self):
        self.make_argmax = self.launcher_config.get('make_argmax', False)

    def process(self, raw, identifiers, frame_meta):
        result = []
        frame_meta = frame_meta or [] * len(identifiers)
        raw_outputs = self._extract_predictions(raw, frame_meta)
        self.select_output_blob(raw_outputs)
        for identifier, output, meta in zip(identifiers, raw_outputs[self.output_blob], frame_meta):
            input_shape = next(iter(meta['input_shape'].values()))
            is_chw = input_shape[1] <= 4
            if len(output.shape) == 2 and len(input_shape) == 4:
                (in_h, in_w) = input_shape[2:] if is_chw else input_shape[1:3]
                if output.shape[0] == in_h * in_w:
                    output = np.resize(output, (in_h, in_w, output.shape[-1]))
                    is_chw = False
            if self.make_argmax:
                argmax_axis = 0 if is_chw else -1
                output = np.argmax(output, axis=argmax_axis)
            if not is_chw and not self.make_argmax and len(output.shape) == 3:
                output = np.transpose(output, (2, 0, 1))
            result.append(SegmentationPrediction(identifier, output))

        return result

    def _extract_predictions(self, outputs_list, meta):
        if 'tiles_shape' not in (meta[-1] or {}):
            return outputs_list[0] if not isinstance(outputs_list, dict) else outputs_list

        self.select_output_blob(outputs_list[0])
        tiles_shapes = [meta['tiles_shape'] for meta in meta]
        restore_output = []
        offset = 0
        for _, image_tiles_shape in enumerate(tiles_shapes):
            next_offset = offset + image_tiles_shape[0] * image_tiles_shape[1]
            image_tiles = [network_output[self.output_blob] for network_output in outputs_list[offset:next_offset]]
            tiles_columns = image_tiles[::image_tiles_shape[0]]
            image = tiles_columns[0]
            for tile_column in tiles_columns[1:]:
                image = np.concatenate((image, tile_column), axis=3)
            restore_output.append(image.squeeze())
            offset = next_offset

        return {self.output_blob: restore_output}


class SegmentationOneClassAdapter(Adapter):
    __provider__ = 'segmentation_one_class'
    prediction_types = (SegmentationPrediction,)

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'threshold': NumberField(
                optional=True, value_type=float, min_value=0.0, default=0.5,
                description='minimal probability threshold for separating predicted class from background'
            )
        })
        return params

    def configure(self):
        self.threshold = self.get_value_from_config('threshold')

    def process(self, raw, identifiers, frame_meta):
        result = []
        frame_meta = frame_meta or [] * len(identifiers)
        raw_outputs = self._extract_predictions(raw, frame_meta)
        self.select_output_blob(raw_outputs)
        for identifier, output in zip(identifiers, raw_outputs[self.output_blob]):
            output = output > self.threshold
            result.append(SegmentationPrediction(identifier, output.astype(np.uint8)))

        return result


class BrainTumorSegmentationAdapter(Adapter):
    __provider__ = 'brain_tumor_segmentation'
    prediction_types = (BrainTumorSegmentationPrediction,)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'make_argmax': BoolField(
                optional=True, default=False, description="Allows to apply argmax operation to output values."
            ),
            'label_order': ListField(
                optional=True, default=[1, 2, 3], value_type=int, validate_values=True,
                description="Specifies order of output labels, according to order of dataset labels"
            ),
            'segmentation_out': StringField(
                optional=True,
                description='Segmentation output layer name. If not provided, first output will be used.'
            )
        })

        return parameters

    def configure(self):
        self.argmax = self.get_value_from_config('make_argmax')
        self.label_order = tuple(self.get_value_from_config('label_order'))
        self.segmentation_out = self.get_value_from_config('segmentation_out')
        if self.segmentation_out:
            self.segmentation_out_bias = self.segmentation_out + '/add_'

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        frame_meta = frame_meta or [] * len(identifiers)
        raw_outputs = self._extract_predictions(raw, frame_meta)
        if self.segmentation_out:
            if not contains_any(raw_outputs, [self.segmentation_out, self.segmentation_out_bias]):
                raise ConfigError('segmentation output not found')
            segm_out = self.segmentation_out if self.segmentation_out in raw_outputs else self.segmentation_out_bias
        else:
            self.select_output_blob(raw_outputs)
            segm_out = self.output_blob
        for identifier, output in zip(identifiers, raw_outputs[segm_out]):
            if self.argmax:
                output = np.argmax(output, axis=0).astype(np.int8)
                output = np.expand_dims(output, axis=0)
            result.append(BrainTumorSegmentationPrediction(identifier, output, self.label_order))

        return result

    def _extract_predictions(self, outputs_list, meta):
        if not (meta[-1] or {}).get('multi_infer', False):
            return outputs_list[0] if not isinstance(outputs_list, dict) else outputs_list

        output_keys = list(outputs_list[0].keys())
        output_map = {}
        for output_key in output_keys:
            output_data = [[output[output_key] for output in outputs_list]]
            output_map[output_key] = output_data

        return output_map


class DUCSegmentationAdapter(Adapter):
    __provider__ = 'duc_segmentation'
    prediction_types = (SegmentationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'ds_rate': NumberField(
                optional=True, default=8, value_type=int, description="Specifies downsample rate."
            ),
            'cell_width': NumberField(
                optional=True, default=2, value_type=int, description="Specifies cell width to extract predictions."
            ),
            'label_num': NumberField(
                optional=True, default=19, value_type=int, description='Specifies number of output label classes.'
            )
        })

        return parameters

    def configure(self):
        self.ds_rate = self.get_value_from_config('ds_rate')
        self.cell_width = self.get_value_from_config('cell_width')
        self.label_num = self.get_value_from_config('label_num')

    def process(self, raw, identifiers, frame_meta):
        result = []
        raw_outputs = self._extract_predictions(raw, frame_meta)
        self.select_output_blob(raw_outputs)
        for identifier, output, meta in zip(identifiers, raw_outputs[self.output_blob], frame_meta):
            _, _, h, w = next(iter(meta.get('input_shape', {'data': (1, 3, 800, 800)}).values()))
            feat_height = math.floor(h / self.ds_rate)
            feat_width = math.floor(w / self.ds_rate)
            labels = output.reshape((self.label_num, 4, 4, feat_height, feat_width))
            labels = np.transpose(labels, (0, 3, 1, 4, 2))
            labels = labels.reshape((self.label_num, int(h / self.cell_width), int(w / self.cell_width)))
            labels = np.transpose(labels, [1, 2, 0])
            labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_LINEAR)
            labels = np.transpose(labels, [2, 0, 1])
            result.append(SegmentationPrediction(identifier, labels))
        return result


class BackgroundMattingAdapter(Adapter):
    __provider__ = 'background_matting'

    def process(self, raw, identifiers, frame_meta):
        result = []
        frame_meta = frame_meta or [] * len(identifiers)
        raw_outputs = self._extract_predictions(raw, frame_meta)
        self.select_output_blob(raw_outputs)
        for identifier, output in zip(identifiers, raw_outputs[self.output_blob]):
            output *= 255
            result.append(BackgroundMattingPrediction(identifier, output.astype(np.uint8)))

        return result
