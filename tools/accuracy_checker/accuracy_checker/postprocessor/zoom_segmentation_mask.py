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
from ..representation import SegmentationAnnotation, SegmentationPrediction
from ..config import NumberField
from ..logging import warning


class ZoomSegMask(Postprocessor):
    """
    Zoom probabilities of segmentation prediction.
    """

    __provider__ = 'zoom_segmentation_mask'

    annotation_types = (SegmentationAnnotation, )
    prediction_types = (SegmentationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'zoom': NumberField(value_type=int, min_value=1, description="Size for zoom operation.")
        })
        return parameters

    def configure(self):
        self.zoom = self.get_value_from_config('zoom')

    def process_image(self, annotation, prediction):
        for annotation_, prediction_ in zip(annotation, prediction):
            prob = prediction_.mask
            if len(prob.shape) == 2:
                warning(
                    'Your prediction mask contains prediction classes instead their probabilities. '
                    'The result can be unpredictable.'
                )
                prob = prob[np.newaxis, :, :]
            channels, prediction_height, prediction_width = prob.shape
            if annotation is not None:
                height, width = annotation_.mask.shape[:2]
            else:
                height, width = prediction_height, prediction_width
            zoom_prob = np.zeros((channels, height, width), dtype=np.float32)
            for c in range(channels):
                for h in range(height):
                    for w in range(width):
                        r0 = h // self.zoom
                        r1 = r0 + 1 if r0 + 1 != prediction_height else r0
                        c0 = w // self.zoom
                        c1 = c0 + 1 if c0 + 1 != prediction_width else c0
                        rt = float(h) / self.zoom - r0
                        ct = float(w) / self.zoom - c0
                        v0 = rt * prob[c, r1, c0] + (1 - rt) * prob[c, r0, c0]
                        v1 = rt * prob[c, r1, c1] + (1 - rt) * prob[c, r0, c1]
                        zoom_prob[c, h, w] = (1 - ct) * v0 + ct * v1
            prediction_.mask = zoom_prob

        return annotation, prediction
