"""
 Copyright (C) 2020-2024 Intel Corporation
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


def scatter(source, classes, axis=1, base=0, value=1):
    shape = [1, 1, *source.shape]
    shape[axis] = classes
    label_map = np.full(shape, base, np.int32)
    ndim = len(shape)
    expanded_index = []
    for i in range(ndim):
        arr = (source if axis == i
               else np.arange(shape[i]).reshape([shape[i] if i == j else 1 for j in range(ndim)]))
        expanded_index.append(arr)
    label_map[tuple(expanded_index)] = value
    return label_map


def preprocess_semantics(semantic_mask, input_size):
    semantic_mask = cv2.resize(semantic_mask, dsize=tuple(input_size[2:]),
                               interpolation=cv2.INTER_NEAREST)
    # create one-hot label map
    semantic_mask = scatter(semantic_mask, classes=input_size[1])

    if len(semantic_mask.shape) == 3:
        return np.expand_dims(semantic_mask, axis=0)
    return semantic_mask


def preprocess_for_seg_model(image, input_size):
    image = cv2.resize(image, dsize=tuple(input_size[2:]), interpolation=cv2.INTER_LINEAR)
    image = np.transpose(image, (2, 0, 1))
    return np.expand_dims(image, axis=0)


def preprocess_image(image, input_size):
    image = cv2.resize(image, dsize=tuple(input_size[2:]), interpolation=cv2.INTER_CUBIC)
    image = np.transpose(image, (2, 0, 1))
    return np.expand_dims(image, axis=0)
