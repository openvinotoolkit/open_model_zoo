"""
 Copyright (c) 2021-2024 Intel Corporation

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
import cv2


def max_central_square_crop(image):
    ''' Makes max-sized central squared crop. '''

    height, width = image.shape[:2]

    if width > height:
        image = image[:, (width - height) // 2:(width - height) // 2 + height]
    else:
        image = image[(height - width) // 2:(height - width) // 2 + width, :]

    return image


def crop_resize(image, input_size):
    ''' Makes max-sized central squared crop and resize to input_size '''
    if input_size[0].is_static and input_size[1].is_static:
        image = max_central_square_crop(image)
        image = cv2.resize(image, (input_size[1].get_length(), input_size[0].get_length()))
    return np.expand_dims(image, axis=0)
