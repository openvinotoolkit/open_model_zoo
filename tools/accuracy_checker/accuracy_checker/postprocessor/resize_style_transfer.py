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
from ..postprocessor import Postprocessor
from ..representation import StyleTransferAnnotation, StyleTransferPrediction
from ..config import NumberField
from ..utils import get_size_from_config


class ResizeStyleTransfer(Postprocessor):
    __provider__ = 'resize_style_transfer'

    annotation_types = (StyleTransferAnnotation, )
    prediction_types = (StyleTransferPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'dst_width': NumberField(
                value_type=int, optional=False, min_value=1, description="Destination width for resizing."
            ),
            'dst_height': NumberField(
                value_type=int, optional=False, min_value=1, description="Destination height for resizing."
            )
        })
        return parameters

    def configure(self):
        self.dst_height, self.dst_width = get_size_from_config(self.config, allow_none=True)

    def process_image(self, annotation, prediction):
        for target in annotation:
            if target is None:
                continue
            data = Image.fromarray(target.value)
            data = data.resize((self.dst_width, self.dst_height), Image.BICUBIC)
            target.value = np.array(data)
        return annotation, prediction
