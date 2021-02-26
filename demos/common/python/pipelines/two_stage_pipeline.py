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

import threading
from collections import deque

from pipelines import AsyncPipeline


def get_boxes(result):
    detections, frame_meta = result
    frame = frame_meta['frame']
    boxes = []
    for detection in detections:
        xmin = max(int(detection.xmin), 0)
        ymin = max(int(detection.ymin), 0)
        xmax = min(int(detection.xmax), frame.shape[1])
        ymax = min(int(detection.ymax), frame.shape[0])
        boxes.append(frame[ymin:ymax, xmin:xmax])
    return boxes


class TwoStagePipeline:
    def __init__(self, ie, encoder_model, decoder_model,
                 en_plugin_config, de_plugin_config,
                 en_device, de_device,
                 en_max_num_requests, de_max_num_requests):

        self.encoder = AsyncPipeline(ie, encoder_model, en_plugin_config, en_device, en_max_num_requests)
        self.decoder = AsyncPipeline(ie, decoder_model, de_plugin_config, de_device, de_max_num_requests)

        self.submitted_frames = deque([])
        self.submitted_enc_results = deque([])

    def is_ready(self):
        return self.encoder.is_ready()

    def await_any(self):
        self.encoder.await_any()

    def await_all(self):
        self.encoder.await_all()

    def has_completed_request(self):
        return self.encoder.has_completed_request()

    def submit_data(self, inputs, id, meta):
        self.submitted_frames.append(id)
        self.encoder.submit_data(inputs, id, meta)

    def get_result(self):
        if not self.submitted_frames:
            return None

        frame_id = self.submitted_frames[0]
        encoder_result = self.encoder.get_result(frame_id)
        if not encoder_result:
            return None

        self.submitted_frames.popleft()
        data = get_boxes(encoder_result)
        self.decoder_result = [None for _ in data]
        self.start_decoder_inference(data)
        return encoder_result, self.decoder_result

    def start_decoder_inference(self, data):
        submission = threading.Thread(target=self.submit_to_decoder, args=(data,))
        postprocess = threading.Thread(target=self.postprocess_all, args=())

        submission.start()
        postprocess.start()

        submission.join()
        postprocess.join()

    def submit_to_decoder(self, data):
        indices = [i for i, _ in enumerate(data)]
        while indices:
            if self.decoder.is_ready():
                id = indices.pop()
                self.submitted_enc_results.append(id)
                self.decoder.submit_data(data[id], id, None)
            else:
                self.decoder.await_any()

    def postprocess_all(self):
        while not self.result_is_ready():
            if self.submitted_enc_results:
                id = self.submitted_enc_results.pop()
                while True:
                    results = self.decoder.get_result(id)
                    if results is not None:
                        self.decoder_result[id] = results[0]
                        break

    def result_is_ready(self):
        return len(self.decoder_result) \
               == len([x for x in self.decoder_result if x is not None])
