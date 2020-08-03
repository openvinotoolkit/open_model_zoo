"""
Copyright (c) 2018-2020 Intel Corporation

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
try:
    from scipy.ndimage import interpolation
except ImportError:
    interpolation = None


def resample(data, shape):
    if interpolation is None:
        raise ValueError('scipy required, please install it')
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
        raise RuntimeError("Since `process_image_with_metadata` is overriden, this method MUST NOT be called")

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
