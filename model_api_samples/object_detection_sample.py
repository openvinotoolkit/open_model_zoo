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

from openvino.model_zoo.model_api.models import DetectionModel

from visualizers import ColorPalette


def draw_detections(frame, detections, palette):
    for detection in detections:
        class_id = int(detection.id)
        color = palette[class_id]
        xmin, ymin, xmax, ymax = detection.get_coords()
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, '{} {:.1%}'.format(detection.str_label, detection.score),
                    (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
    return frame


def main():
    if len(sys.argv) != 2:
        raise RuntimeError(f'Usage: {sys.argv[0]} <path_to_image>')
    model = DetectionModel.create_model('yolo-v4-tf')
    image = cv2.imread(sys.argv[1])
    if image is None:
        raise RuntimeError('Failed to read the image')
    objects = model(image)
    draw_detections(image, objects, ColorPalette(n=100))
    cv2.imshow('Detection Results', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
