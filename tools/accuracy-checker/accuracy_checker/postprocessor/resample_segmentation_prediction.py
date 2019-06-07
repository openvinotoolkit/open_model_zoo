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
from scipy.ndimage import interpolation
from .postprocessor import Postprocessor
from ..representation import BrainTumorSegmentationPrediction, BrainTumorSegmentationAnnotation


def resample(data, shape):
    if len(data.shape) != len(shape):
        raise RuntimeError('Dimensions of input array and shape are different. Resampling is impossible.')
    factor = [float(o) / i for i, o in zip(data.shape, shape)]
    return interpolation.zoom(data, zoom=factor, order=1)


class SegmentationPredictionResample(Postprocessor):
    __provider__ = "segmentation_prediction_resample"

    prediction_types = (BrainTumorSegmentationPrediction, )
    annotation_types = (BrainTumorSegmentationAnnotation, )

    def process_image(self, annotation, prediction):
        if not len(annotation) == len(prediction) == 1:
            raise RuntimeError('Postprocessor {} does not support multiple annotation and/or prediction.'
                               .format(self.__provider__))

        if not annotation[0].box:
            raise ValueError('Postprocessor {} not found bounding box.'.format(self.__provider__))

        annotation_ = annotation[0]
        prediction_ = prediction[0]

        low = annotation_.box[0, :]
        high = annotation_.box[1, :]
        diff = (high - low).astype(np.int32)

        annotation_shape = annotation_.mask.shape
        prediction_shape = prediction_.mask.shape

        image_shape = annotation_shape[len(annotation_shape)-3:len(annotation_shape)]
        box_shape = (diff[0], diff[1], diff[2])

        label = np.zeros(shape=(prediction_shape[0],) + image_shape)

        label[:, low[0]:high[0], low[1]:high[1], low[2]:high[2]] = resample(prediction_.mask,
                                                                            (prediction_shape[0],) + box_shape)

        prediction[0].mask = label

        return annotation, prediction
