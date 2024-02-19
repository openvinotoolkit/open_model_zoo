"""
 Copyright (C) 2018-2024 Intel Corporation

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

from .segmentation import SegmentationModel


class MonoDepthModel(SegmentationModel):
    __model__ = 'MonoDepth'

    def postprocess(self, outputs, meta):
        result = outputs[self.output_blob_name].squeeze()
        input_image_height = meta['original_shape'][0]
        input_image_width = meta['original_shape'][1]

        result = cv2.resize(result, (input_image_width, input_image_height), interpolation=cv2.INTER_CUBIC)

        disp_min = result.min()
        disp_max = result.max()
        if disp_max - disp_min > 1e-6:
            result = (result - disp_min) / (disp_max - disp_min)
        else:
            result.fill(0.5)

        return result
