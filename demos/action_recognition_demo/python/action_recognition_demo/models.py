"""
 Copyright (c) 2020-2024 Intel Corporation

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

from collections import deque
from itertools import cycle
import sys

import logging as log
import cv2
import numpy as np

from openvino import AsyncInferQueue


def center_crop(frame, crop_size):
    img_h, img_w, _ = frame.shape

    x0 = int(round((img_w - crop_size[0]) / 2.))
    y0 = int(round((img_h - crop_size[1]) / 2.))
    x1 = x0 + crop_size[0]
    y1 = y0 + crop_size[1]

    return frame[y0:y1, x0:x1, ...]


def adaptive_resize(frame, dst_size):
    h, w, _ = frame.shape
    scale = dst_size / min(h, w)
    ow, oh = int(w * scale), int(h * scale)

    if ow == w and oh == h:
        return frame
    return cv2.resize(frame, (ow, oh))


def preprocess_frame(frame, size=224, crop_size=224, chw_layout=True):
    frame = adaptive_resize(frame, size)
    frame = center_crop(frame, (crop_size, crop_size))
    if chw_layout:
        frame = frame.transpose((2, 0, 1))  # HWC -> CHW

    return frame


class AsyncWrapper:
    def __init__(self, ie_model, num_requests):
        self.model = ie_model
        self.num_requests = num_requests
        self._result_ready = False
        self._req_ids = cycle(range(num_requests))
        self._result_ids = cycle(range(num_requests))
        self._frames = deque(maxlen=num_requests)

    def infer(self, model_input, frame=None):
        """Schedule current model input to infer, return last result"""
        next_req_id = next(self._req_ids)
        self.model.async_infer(model_input, next_req_id)

        last_frame = self._frames[0] if self._frames else frame

        self._frames.append(frame)
        if next_req_id == self.num_requests - 1:
            self._result_ready = True

        if self._result_ready:
            result_req_id = next(self._result_ids)
            result = self.model.wait_request(result_req_id)
            return result, last_frame
        else:
            return None, None


class IEModel:
    def __init__(self, model_path, core, target_device, num_requests, model_type):
        log.info('Reading {} model {}'.format(model_type, model_path))
        self.model = core.read_model(model_path)
        if len(self.model.inputs) != 1:
            log.error("Demo supports only models with 1 input")
            sys.exit(1)

        if len(self.model.outputs) != 1:
            log.error("Demo supports only models with 1 output")
            sys.exit(1)

        self.outputs = {}
        compiled_model = core.compile_model(self.model, target_device)
        self.output_tensor = compiled_model.outputs[0]
        self.input_name = self.model.inputs[0].get_any_name()
        self.input_shape = self.model.inputs[0].shape

        self.num_requests = num_requests
        self.infer_queue = AsyncInferQueue(compiled_model, num_requests)
        self.infer_queue.set_callback(self.completion_callback)
        log.info('The {} model {} is loaded to {}'.format(model_type, model_path, target_device))

    def completion_callback(self, infer_request, id):
        self.outputs[id] = infer_request.results[self.output_tensor]

    def async_infer(self, frame, req_id):
        input_data = {self.input_name: frame}
        self.infer_queue.start_async(input_data, req_id)

    def wait_request(self, req_id):
        self.infer_queue[req_id].wait()
        return self.outputs.pop(req_id, None)

    def cancel(self):
        for ireq in self.infer_queue:
            ireq.cancel()


class DummyDecoder:
    def __init__(self, num_requests=2):
        self.num_requests = num_requests
        self.requests = {}

    @staticmethod
    def _average(model_input):
        return np.mean(model_input, axis=1)

    def async_infer(self, model_input, req_id):
        self.requests[req_id] = self._average(model_input)

    def wait_request(self, req_id):
        assert req_id in self.requests
        return self.requests.pop(req_id)

    def cancel(self):
        pass
