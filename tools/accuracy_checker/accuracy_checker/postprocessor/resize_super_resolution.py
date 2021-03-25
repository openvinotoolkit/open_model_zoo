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
from PIL import Image
from .postprocessor import Postprocessor
from ..representation import SuperResolutionPrediction, SuperResolutionAnnotation
from ..config import NumberField, StringField
from ..utils import get_size_from_config


class ResizeSuperResolution(Postprocessor):
    __provider__ = 'resize_super_resolution'

    annotation_types = (SuperResolutionAnnotation, )
    prediction_types = (SuperResolutionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'dst_width': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination width for resizing."
            ),
            'dst_height': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination height for resizing."
            ),
            'target': StringField(choices=['annotation', 'prediction'], optional=True, default='prediction'),
        })

        return parameters

    def configure(self):
        if Image is None:
            raise ValueError('{} requires pillow, please install it'.format(self.__provider__))
        self.dst_height, self.dst_width = get_size_from_config(self.config, allow_none=True)
        self.target = self.get_value_from_config('target')

    def process_image(self, annotation, prediction):
        for annotation_, prediction_ in zip(annotation, prediction):
            if annotation_ is None and self.target != 'prediction':
                continue
            target_height = (
                self.dst_height or
                (annotation_.value.shape[0] if annotation_ is not None else self.image_size[0])
            )
            target_width = (
                self.dst_width or
                (annotation_.value.shape[1] if annotation_ is not None else self.image_size[1])
            )
            data = Image.fromarray(prediction_.value if self.target == 'prediction' else annotation_.value)
            data = data.resize((target_width, target_height), Image.BICUBIC)
            if self.target == 'prediction':
                prediction_.value = np.array(data)
            else:
                annotation_.value = np.array(data)

        return annotation, prediction
