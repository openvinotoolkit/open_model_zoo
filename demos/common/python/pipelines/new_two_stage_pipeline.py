"""
 Copyright (C) 2021 Intel Corporation

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

import logging
from collections import deque
from models.utils import preprocess_output


class newTwoStagePipeline:
    def __init__(self, ie, detector, recognizer, td_plugin_config, tr_plugin_config,
                 td_device, tr_device, td_num_requests, tr_num_requests):
        self.logger = logging.getLogger()
        self.detector = detector
        self.recognizer = recognizer

        self.logger.info('Loading Text Detection network to {} plugin...'.format(td_device))
        self.exec_net_detector = ie.load_network(network=self.detector.net, device_name=td_device,
                                                 config=td_plugin_config, num_requests=td_num_requests)
        self.logger.info('Loading Text Recognition network to {} plugin...'.format(tr_device))
        self.exec_net_recognizer = ie.load_network(network=self.recognizer.net, device_name=tr_device,
                                                   config=tr_plugin_config, num_requests=tr_num_requests)

        detector_req_id = [id for id in range(td_num_requests)]
        recognizer_req_id = [id for id in range(tr_num_requests)]

        self.empty_detector_req_id = deque(detector_req_id)
        self.empty_recognizer_req_id = deque(recognizer_req_id)

        self.processed_detector_req_id = deque([])
        self.processed_recognizer_req_id = deque([])

        self.detector_meta = {req_id : None for req_id in detector_req_id}     # [frame_id, preprocess_meta, meta]
        self.recognizer_meta = {req_id : None for req_id in recognizer_req_id} # [box_id, preprocess_meta]

        self.detector_result = {} # {frame_id: result}, it returns to the user
        self.detector_boxes = {}  # {frame_id: [boxes, boxes_number]}

        self.recognizer_result = {}

    def get_result(self, id):
        self.check_detector_status()
        if id not in self.detector_boxes:
            return None

        if self.result_is_ready(id):
            self.detector_boxes.pop(id)
            return self.detector_result.pop(id), self.recognizer_result.pop(id)

        self.check_recognizer_status()

        if self.is_recognizer_ready():
            boxes = self.detector_boxes[id][0]
            if boxes:
                box = boxes.pop()
                box_id = len(self.detector_boxes[id][0])
                self.submit_recognizer_data(box, box_id, id)
            else: # current id boxes are processed, can sumbit boxes for next id
                next_id = id + 1
                if next_id not in self.detector_boxes:
                    return None
                boxes = self.detector_boxes[next_id][0]
                if boxes:
                    box = boxes.pop()
                    box_id = len(self.detector_boxes[next_id][0])
                    self.submit_recognizer_data(box, box_id, next_id)
        return None

    def result_is_ready(self, id):
        return self.detector_boxes[id][1] \
               == len([x for x in self.recognizer_result[id] if x is not None])

    def check_recognizer_status(self):
        i = 0
        while i < len(self.processed_recognizer_req_id):
            req_id = self.processed_recognizer_req_id[i]
            if self.exec_net_recognizer.requests[req_id].wait(0) == 0:
                result, box_id, frame_id = self.get_recognizer_result(req_id)
                self.recognizer_result[frame_id][box_id] = result
                del self.processed_recognizer_req_id[i]
                self.empty_recognizer_req_id.append(req_id)
            else:
                i += 1

    def check_detector_status(self):
        i = 0
        while i < len(self.processed_detector_req_id):
            req_id = self.processed_detector_req_id[i]
            if self.exec_net_detector.requests[req_id].wait(3) == 0:
                result, id = self.get_detector_result(req_id)
                boxes = preprocess_output(result)
                self.detector_result[id] = result
                self.detector_boxes[id] = (boxes, len(boxes))
                self.recognizer_result[id] = [None for _ in range(len(boxes))]
                del self.processed_detector_req_id[i]
                self.empty_detector_req_id.append(req_id)
            else:
                i += 1

    def get_detector_result(self, request_id):
        request = self.exec_net_detector.requests[request_id]
        frame_id, preprocess_meta, meta = self.get_detector_meta(request_id)
        raw_result = {key: blob.buffer for key, blob in request.output_blobs.items()}
        return (self.detector.postprocess(raw_result, preprocess_meta), meta), frame_id

    def get_recognizer_result(self, request_id):
        request = self.exec_net_recognizer.requests[request_id]
        box_id, frame_id, preprocess_meta = self.get_recognizer_meta(request_id)
        raw_result = {key: blob.buffer for key, blob in request.output_blobs.items()}
        return self.recognizer.postprocess(raw_result, preprocess_meta), box_id, frame_id

    def get_detector_meta(self, request_id):
        meta = self.detector_meta[request_id]
        self.detector_meta[request_id] = None
        return meta

    def get_recognizer_meta(self, request_id):
        meta = self.recognizer_meta[request_id]
        self.recognizer_meta[request_id] = None
        return meta

    def submit_data(self, inputs, id, meta):
        request_id = self.empty_detector_req_id.popleft()
        request = self.exec_net_detector.requests[request_id]

        inputs, preprocessing_meta = self.detector.preprocess(inputs)

        self.processed_detector_req_id.append(request_id)
        self.detector_meta[request_id] = (id, preprocessing_meta, meta)

        request.async_infer(inputs=inputs)

    def submit_recognizer_data(self, inputs, box_id, frame_id):
        request_id = self.empty_recognizer_req_id.popleft()
        request = self.exec_net_recognizer.requests[request_id]

        inputs, preprocessing_meta = self.recognizer.preprocess(inputs)

        self.processed_recognizer_req_id.append(request_id)
        self.recognizer_meta[request_id] = (box_id, frame_id, preprocessing_meta)

        request.async_infer(inputs=inputs)

    def is_ready(self):
        return len(self.empty_detector_req_id) != 0

    def is_recognizer_ready(self):
        return len(self.empty_recognizer_req_id) != 0

    def await_all(self):
        for request in self.exec_net_detector.requests:
            request.wait()

    def has_completed_request(self):
        return len(self.processed_detector_req_id) or len(self.processed_recognizer_req_id)
