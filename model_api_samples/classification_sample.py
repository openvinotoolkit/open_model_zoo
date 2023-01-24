#!/usr/bin/env python3
"""
 Copyright (C) 2018-2022 Intel Corporation

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

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/model_tools/src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'demos/common/python'))

from openvino.model_zoo.model_api.models import Classification


def draw_labels(frame, classifications):
    class_label = ""
    if classifications:
        class_label = classifications[0][1]
    font_scale = 0.7
    label_height = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)[0][1]
    initial_labels_pos =  frame.shape[0] - label_height * (int(1.5 * len(classifications)) + 1)

    if (initial_labels_pos < 0):
        initial_labels_pos = label_height
        print('Too much labels to display on this frame, some will be omitted')
    offset_y = initial_labels_pos

    header = "Label:     Score:"
    label_width = cv2.getTextSize(header, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)[0][0]
    cv2.putText(frame, header, (frame.shape[1] - label_width, offset_y),
        cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 0, 0), 2)

    for idx, class_label, score in classifications:
        label = '{}. {}    {:.2f}'.format(idx, class_label, score)
        label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)[0][0]
        offset_y += int(label_height * 1.5)
        cv2.putText(frame, label, (frame.shape[1] - label_width, offset_y),
            cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 0, 0), 2)
    return frame


def main():
    if len(sys.argv) != 2:
        raise RuntimeError(f'Usage: {sys.argv[0]} <path_to_image>')
    classifier = Classification.create_model('resnet-18-pytorch')
    image = cv2.imread(sys.argv[1])
    if image is None:
        raise RuntimeError('Failed to read the image')
    classifications = classifier(image)
    image = draw_labels(image, classifications)
    cv2.imshow('Detection Results', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
