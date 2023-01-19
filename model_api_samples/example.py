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

sys.path.append(str(Path(__file__).resolve().parent / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parent / 'common/python/openvino/model_zoo'))

from model_api.models import DetectionModel
from model_api.pipelines import get_user_config
from model_api.adapters import create_core, OpenvinoAdapter

from visualizers import ColorPalette


def draw_detections(frame, detections, palette, labels):
    for detection in detections:
        class_id = int(detection.id)
        color = palette[class_id]
        det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
        xmin, ymin, xmax, ymax = detection.get_coords()
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, '{} {:.1%}'.format(det_label, detection.score),
                    (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
    return frame


def main():
    plugin_config = get_user_config('CPU', flags_nstreams='1', flags_nthreads=None)
    model_adapter = OpenvinoAdapter(create_core(), 'yolo-v4-tf',
        device='CPU', plugin_config=plugin_config, max_num_requests=1, model_parameters={'input_layouts': None},
        precision='FP16', download_dir=None, cache_dir=None)
    configuration = {
        'resize_type': 'fit_to_window_letterbox',
        'mean_values': None,
        'scale_values': None,
        'reverse_input_channels': None,
        'path_to_labels': None,
        'confidence_threshold': 0.5
    }
    detector = DetectionModel.create_model(model_adapter, configuration=configuration, preload=True)
    image = cv2.imread('/home/wov/Pictures/dog-0000.jpg')
    objects = detector(image)[0]
    draw_detections(image, objects, ColorPalette(n=100), labels=None)
    cv2.imshow('Detection Results', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
