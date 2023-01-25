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


def main():
    if len(sys.argv) != 2:
        raise RuntimeError(f'Usage: {sys.argv[0]} <path_to_image>')
    model = DetectionModel.create_model('yolo-v4-tf', max_num_requests=0)
    image = cv2.imread(sys.argv[1])
    if image is None:
        raise RuntimeError('Failed to read the image')

    INFERENCE_NUMBER = 10
    results = [False for _ in range(INFERENCE_NUMBER)]  # container for results
    def callback(result, userdata):
        print(f"Number: {userdata}, Result: {result}")
        results[userdata] = True

    model.set_callback(callback)
    ## Run parallel inference
    for i in range(INFERENCE_NUMBER):
        model.infer_async(image, i)
    model.await_all()
    assert(all(results))



if __name__ == '__main__':
    main()
