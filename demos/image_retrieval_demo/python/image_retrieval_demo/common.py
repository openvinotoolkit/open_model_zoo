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

import os
import cv2
import numpy as np


def max_central_square_crop(image):
    ''' Makes max-sized central squared crop. '''

    height, width = image.shape[:2]

    if width > height:
        image = image[:, (width - height) // 2:(width - height) // 2 + height]
    else:
        image = image[(height - width) // 2:(height - width) // 2 + width, :]

    return image


def fit_to_max_size(image, max_size):
    ''' Fits input image to max_size. '''

    if image.shape[0] > max_size or image.shape[1] > max_size:
        if image.shape[0] > image.shape[1]:
            image = cv2.resize(image, (int(image.shape[1] / (image.shape[0] / max_size)), max_size))
        else:
            image = cv2.resize(image, (max_size, int(image.shape[0] / (image.shape[1] / max_size))))

    return image


def crop_resize(image, input_size):
    ''' Makes max-sized central crop, resizes to input_size '''

    image = max_central_square_crop(image)
    image = cv2.resize(image, (input_size, input_size))
    image = np.expand_dims(image, axis=0)
    return image


def central_crop(image, divide_by, shift):
    ''' Makes central crops dividing input image by number of equal cells. '''

    height, width = image.shape[0:2]
    image = image[height // divide_by * shift: height // divide_by * (divide_by - shift),
                  width // divide_by * shift: width // divide_by * (divide_by - shift)]
    return image


def from_list(path, multiple_images_per_label=True):
    ''' Loads images list. '''

    impaths = []
    labels = []
    is_real = []

    text_label_to_class_id = {}

    uniques_labels = set()

    root = os.path.dirname(os.path.abspath(path))

    with open(path) as opened_file:
        for line in opened_file.readlines():
            line = line.strip().split(' ')
            if len(line) == 2:
                impath, label = line
                real = False
            else:
                impath, label, real = line
                real = real.lower() == 'r'

            text_label_to_class_id[os.path.basename(impath).split('.')[0]] = int(label)

            if not multiple_images_per_label and label in uniques_labels:
                continue

            uniques_labels.add(label)

            is_real.append(real)
            impaths.append(os.path.join(root, impath))
            labels.append(int(label))

    return impaths, labels, is_real, text_label_to_class_id
