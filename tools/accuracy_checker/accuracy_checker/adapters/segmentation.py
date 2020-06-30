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

import numpy as np
from ..adapters import Adapter
from ..representation import SegmentationPrediction, BrainTumorSegmentationPrediction
from ..config import ConfigValidator, BoolField, ListField, NumberField


class SegmentationAdapter(Adapter):
    __provider__ = 'segmentation'
    prediction_types = (SegmentationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'make_argmax': BoolField(
                optional=True, default=False, description="Allows to apply argmax operation to output values."
            )
        })
        return parameters

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT)

    def configure(self):
        self.make_argmax = self.launcher_config.get('make_argmax', False)

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        frame_meta = frame_meta or [] * len(identifiers)
        raw_outputs = self._extract_predictions(raw, frame_meta)
        for identifier, output in zip(identifiers, raw_outputs[self.output_blob]):
            if self.make_argmax:
                output = np.argmax(output, axis=0)
            result.append(SegmentationPrediction(identifier, output))

        return result

    def _extract_predictions(self, outputs_list, meta):
        if not 'tiles_shape' in (meta[-1] or {}):
            return outputs_list[0] if not isinstance(outputs_list, dict) else outputs_list

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
    prediction_types = (SegmentationPrediction, )

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

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        frame_meta = frame_meta or [] * len(identifiers)
        raw_outputs = self._extract_predictions(raw, frame_meta)
        for identifier, output in zip(identifiers, raw_outputs[self.output_blob]):
            output = output > self.threshold
            result.append(SegmentationPrediction(identifier, output.astype(np.uint8)))

        return result


class BrainTumorSegmentationAdapter(Adapter):
    __provider__ = 'brain_tumor_segmentation'
    prediction_types = (BrainTumorSegmentationPrediction, )

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
            )
        })

        return parameters

    def configure(self):
        self.argmax = self.get_value_from_config('make_argmax')
        self.label_order = tuple(self.get_value_from_config('label_order'))

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        frame_meta = frame_meta or [] * len(identifiers)
        raw_outputs = self._extract_predictions(raw, frame_meta)
        for identifier, output in zip(identifiers, raw_outputs[self.output_blob]):
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
