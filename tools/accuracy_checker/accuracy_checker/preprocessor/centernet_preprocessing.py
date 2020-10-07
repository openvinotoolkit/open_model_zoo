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

import cv2
import numpy as np

from ..config import NumberField
from ..preprocessor import Preprocessor
from ..utils import get_size_from_config


class CenterNetAffineTransformation(Preprocessor):
    __provider__ = 'centernet_affine_transform'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'size': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination size for image."
            ),
            'dst_width': NumberField(
                value_type=int, optional=False, min_value=1, description="Destination width for image."
            ),
            'dst_height': NumberField(
                value_type=int, optional=False, min_value=1, description="Destination height for image."
            ),
            'scale': NumberField(
                value_type=int, optional=True, default=1,
                description="Scale for input image"
            )
        })

        return parameters

    def configure(self):
        self.input_height, self.input_width = get_size_from_config(self.config)
        self.scale = self.get_value_from_config('scale')

    @staticmethod
    def get_affine_transform(center, scale, rot, output_size, inv=0):

        def get_dir(src_point, rot_rad):
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            src_result = [0, 0]
            src_result[0] = src_point[0] * cs - src_point[1] * sn
            src_result[1] = src_point[0] * sn + src_point[1] * cs
            return src_result

        def get_3rd_point(a, b):
            direct = a - b
            return b + np.array([-direct[1], direct[0]], dtype=np.float32)

        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)

        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w, dst_h = output_size

        rot_rad = np.pi * rot / 180
        src_dir = get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

        dst = np.zeros((3, 2), dtype=np.float32)
        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :], src[1, :] = center, center + src_dir
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
        src[2:, :] = get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def process(self, image, annotation_meta=None):
        data = image.data
        height, width = data.shape[0:2]
        new_height = height * self.scale
        new_width = width * self.scale
        center = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
        scale = max(height, width) * 1.0

        trans_input = self.get_affine_transform(center, scale, 0, [self.input_width, self.input_height])
        resized_image = cv2.resize(data, (new_width, new_height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (self.input_width, self.input_height),
            flags=cv2.INTER_LINEAR)

        image.data = inp_image
        return image
