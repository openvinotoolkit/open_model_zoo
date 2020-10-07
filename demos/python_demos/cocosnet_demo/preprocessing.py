"""
 Copyright (C) 2020 Intel Corporation
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


def scatter(source):
    out = np.zeros((1, 151, 256, 256))
    h, w = source.shape
    for i in range(h):
        for j in range(w):
            out[0][source[i][j]][i][j] = 1
    return out


def preprocess_with_semantics(semantic_mask):
    semantic_mask = cv2.resize(semantic_mask, dsize=(256, 256),
                               interpolation=cv2.INTER_NEAREST)
    # create one-hot label map
    semantic_mask = scatter(semantic_mask).astype(np.int)
    return semantic_mask


def normalization(img, mean=[127.5], scale=[127.5]):
    channels = img.shape[0]
    img = img.astype(np.float)
    if len(mean) == 1:
        mean = channels * mean
        scale = channels * scale
    for channel in range(channels):
        channel_val = np.subtract(img[channel, :, :], mean[channel])
        channel_val /= scale[channel]
        img[channel, :, :] = channel_val
    return img


def preprocess_for_seg_model(image):
    image = cv2.resize(image, dsize=(320, 320), interpolation=cv2.INTER_LINEAR)
    image = np.transpose(image, (2, 0, 1))
    return image


def preprocess_with_images(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    image = np.transpose(image, (2, 0, 1))
    image = normalization(image, mean=[127.5], scale=[127.5])
    return image
