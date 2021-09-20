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

import numpy as np
from .postprocessor import Postprocessor
from ..config import ConfigError, BoolField, ListField, NumberField
from ..representation import BrainTumorSegmentationPrediction, BrainTumorSegmentationAnnotation
from ..utils import UnsupportedPackage
try:
    from scipy.ndimage import interpolation
except ImportError as import_error:
    interpolation = UnsupportedPackage("scipy", import_error.msg)


def resample(data, shape):
    if isinstance(interpolation, UnsupportedPackage):
        interpolation.raise_error("segmentation_prediction_resample")
    if len(data.shape) != len(shape):
        raise RuntimeError('Dimensions of input array and shape are different. Resampling is impossible.')
    factor = [float(o) / i for i, o in zip(data.shape, shape)]
    return interpolation.zoom(data, zoom=factor, order=1)


class SegmentationPredictionResample(Postprocessor):
    __provider__ = "segmentation_prediction_resample"

    prediction_types = (BrainTumorSegmentationPrediction, )
    annotation_types = (BrainTumorSegmentationAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'make_argmax': BoolField(optional=True, default=False,
                                     description="Applies argmax operation for prediction")
        })
        return parameters

    def configure(self):
        self.make_argmax = self.config.get('make_argmax')

    def process_image(self, annotations, predictions):
        raise RuntimeError("Since `process_image_with_metadata` is overridden, this method MUST NOT be called")

    def process_image_with_metadata(self, annotation, prediction, image_metadata=None):
        if not len(annotation) == len(prediction) == 1:
            raise RuntimeError('Postprocessor {} does not support multiple annotation and/or prediction.'
                               .format(self.__provider__))

        if annotation is not None and annotation[0].box:
            box = annotation[0].box
        elif image_metadata['box'] is not None:
            box = image_metadata['box']
        else:
            raise ValueError('Postprocessor {} not found bounding box.'.format(self.__provider__))

        annotation_ = annotation[0]
        prediction_ = prediction[0]

        low = box[0, :]
        high = box[1, :]
        diff = (high - low).astype(np.int32)

        annotation_shape = annotation_.mask.shape if annotation_ is not None else prediction_.mask.shape
        prediction_shape = prediction_.mask.shape

        image_shape = annotation_shape[-3:]
        box_shape = (diff[0], diff[1], diff[2])

        label = np.zeros(shape=(prediction_shape[0],) + image_shape)

        label[:, low[0]:high[0], low[1]:high[1], low[2]:high[2]] = resample(
            prediction_.mask, (prediction_shape[0],) + box_shape
        )

        if self.make_argmax:
            label = np.argmax(label, axis=0).astype(np.int8)
            label = np.expand_dims(label, axis=0)

        prediction[0].mask = label

        return annotation, prediction


class TransformBratsPrediction(Postprocessor):
    __provider__ = 'transform_brats_prediction'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'order': ListField(
                value_type=NumberField(value_type=int, min_value=0), validate_values=True,
                description="Specifies channel order of filling"
            ),
            'values': ListField(
                value_type=int, validate_values=True,
                description="Specifies values for each channel according to new order"
            )
        })
        return parameters

    def configure(self):
        self.order = self.get_value_from_config('order')
        self.values = self.get_value_from_config('values')
        if len(self.order) != len(self.values):
            raise ConfigError('Length of "order" and "values" must be the same')

    def process_image(self, annotation, prediction):
        for target in prediction:
            data = target.mask

            result = np.zeros(shape=data.shape[1:], dtype=np.int8)

            label = data > 0.5
            for i, value in zip(self.order, self.values):
                result[label[i, :, :, :]] = value

            result = np.expand_dims(result, axis=0)

            target.mask = result

        return annotation, prediction


class RemoveBratsPredictionPadding(Postprocessor):
    __provider__ = 'remove_brats_prediction_padding'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'make_argmax': BoolField(
                optional=True, default=False, description="Allows to apply argmax operation to output values."
            )
        })
        return parameters

    def configure(self):
        self.make_argmax = self.get_value_from_config('make_argmax')

    def process_image(self, annotation, prediction):
        raise RuntimeError("Since `process_image_with_metadata` is overridden, this method MUST NOT be called")

    def process_image_with_metadata(self, annotation, prediction, image_metadata=None):
        raw_shape = image_metadata.get('size_after_cropping')
        if not raw_shape:
            raise ValueError("No 'size_after_cropping' in metadata")

        for target in prediction:

            # Remove padding
            padded_shape = target.mask.shape[1:]
            pad_before = [(p - r) // 2 for p, r in zip(padded_shape, raw_shape)]
            pad_after = [-(p - r - b) for p, r, b in zip(padded_shape, raw_shape, pad_before)]
            result = target.mask[:, pad_before[0]:pad_after[0], pad_before[1]:pad_after[1], pad_before[2]:pad_after[2]]

            # Undo cropping
            bbox = image_metadata.get('crop_bbox')
            if not bbox:
                raise ValueError("No 'crop_bbox' in metadata")

            original_size = image_metadata.get('original_size_of_raw_data')
            if original_size is None:
                raise ValueError("No 'original_size_of_raw_data' in metadata")

            label = np.zeros(shape=([target.mask.shape[0]] + list(image_metadata['original_size_of_raw_data'])))
            label[:, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = result
            target.mask = label

            # Apply argmax
            if self.make_argmax:
                target.mask = np.argmax(target.mask, axis=0)
                target.mask = np.expand_dims(target.mask, axis=0)

            # Align with annotation shape
            target.mask = np.transpose(target.mask, (1, 2, 3, 0))

        return annotation, prediction
