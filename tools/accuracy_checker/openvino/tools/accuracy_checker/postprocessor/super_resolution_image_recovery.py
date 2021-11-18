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

import cv2
import numpy as np
from PIL import Image

from ..config import StringField, NumberField, ConfigError
from .postprocessor import Postprocessor
from ..representation import (
    SuperResolutionPrediction, SuperResolutionAnnotation, ImageProcessingAnnotation, ImageProcessingPrediction
)
from ..utils import get_size_from_config


class SRImageRecovery(Postprocessor):
    __provider__ = 'sr_image_recovery'

    annotation_types = (SuperResolutionAnnotation, ImageProcessingAnnotation)
    prediction_types = (SuperResolutionPrediction, ImageProcessingPrediction)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'target_color': StringField(optional=True, choices=['bgr', 'rgb'], default='rgb'),
            'dst_width': NumberField(
                value_type=int, optional=True, min_value=1, description="Width of model input before recovering"
            ),
            'dst_height': NumberField(
                value_type=int, optional=True, min_value=1, description="Height of model input before recovering"
            ),
            'size': NumberField(
                value_type=int, optional=True, min_value=1,
                description="Size of model input for both dimensions (height and width) before recovering"
            )
        })
        return parameters

    def configure(self):
        self.color = cv2.COLOR_YCrCb2BGR if self.get_value_from_config('target_color') == 'bgr' else cv2.COLOR_YCrCb2RGB
        self.input_height, self.input_width = get_size_from_config(self.config)
        if not self.input_height or not self.input_width:
            raise ConfigError('Neither size nor dst_width and dst_height are correct. Please check config values')

    def process_image(self, annotation, prediction):
        for annotation_, prediction_ in zip(annotation, prediction):
            data = Image.fromarray(annotation_.value, 'RGB')
            data = data.resize((self.input_width, self.input_height), Image.BICUBIC)
            data = np.array(data)
            ycrcbdata = cv2.cvtColor(data, cv2.COLOR_RGB2YCrCb)
            cr = ycrcbdata[:, :, 1]
            cb = ycrcbdata[:, :, 2]
            h, w, _ = prediction_.value.shape
            cr = Image.fromarray(np.uint8(cr), mode='L')
            cb = Image.fromarray(np.uint8(cb), mode='L')
            cr = cr.resize((w, h), Image.BICUBIC)
            cb = cb.resize((w, h), Image.BICUBIC)
            cr = np.expand_dims(np.array(cr).astype(np.uint8), axis=-1)
            cb = np.expand_dims(np.array(cb).astype(np.uint8), axis=-1)
            ycrcb = np.concatenate([prediction_.value, cr, cb], axis=2)
            prediction_.value = cv2.cvtColor(ycrcb, self.color)
        return annotation, prediction


class ColorizationLABRecovery(Postprocessor):
    __provider__ = 'colorization_recovery'
    annotation_types = (ImageProcessingAnnotation, )
    prediction_types = (ImageProcessingPrediction, )

    def process_image(self, annotation, prediction):
        for ann, pred in zip(annotation, prediction):
            target = ann.value
            h, w = pred.value.shape[:2]
            r_target = cv2.resize(target, (w, h)).astype(np.float32)
            target_l = cv2.cvtColor(r_target / 255, cv2.COLOR_BGR2LAB)[:, :, 0]
            pred_ab = pred.value
            out_lab = np.concatenate((target_l[:, :, np.newaxis], pred_ab), axis=2)
            result_bgr = cv2.cvtColor(out_lab, cv2.COLOR_Lab2BGR) * 255
            pred.value = result_bgr.astype(np.uint8)
        return annotation, prediction
