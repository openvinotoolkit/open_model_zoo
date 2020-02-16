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

import cv2
import numpy as np

from asl_recognition_demo.common import IEModel


class ActionRecognizer(IEModel):
    """ Class that is used to work with action recognition model. """

    def __init__(self, model_path, device, ie_core, num_requests, img_scale, num_classes):
        """Constructor"""

        super().__init__(model_path, device, ie_core, num_requests)

        _, _, t, h, w = self.input_size
        self.input_height = h
        self.input_width = w
        self.input_length = t

        self.img_scale = img_scale
        self.num_test_classes = num_classes

    @staticmethod
    def _convert_to_central_roi(src_roi, input_height, input_width, img_scale):
        """Extracts from the input ROI the central square part with specified side size"""

        src_roi_height, src_roi_width = src_roi[3] - src_roi[1], src_roi[2] - src_roi[0]
        src_roi_center_x = 0.5 * (src_roi[0] + src_roi[2])
        src_roi_center_y = 0.5 * (src_roi[1] + src_roi[3])

        height_scale = float(input_height) / float(img_scale)
        width_scale = float(input_width) / float(img_scale)
        assert height_scale < 1.0
        assert width_scale < 1.0

        min_roi_size = min(src_roi_height, src_roi_width)
        trg_roi_height = int(height_scale * min_roi_size)
        trg_roi_width = int(width_scale * min_roi_size)

        trg_roi = [int(src_roi_center_x - 0.5 * trg_roi_width),
                   int(src_roi_center_y - 0.5 * trg_roi_height),
                   int(src_roi_center_x + 0.5 * trg_roi_width),
                   int(src_roi_center_y + 0.5 * trg_roi_height)]

        return trg_roi

    def _process_image(self, input_image, roi):
        """Converts input image according to model requirements"""

        cropped_image = input_image[roi[1]:roi[3], roi[0]:roi[2]]
        resized_image = cv2.resize(cropped_image, (self.input_width, self.input_height))
        out_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        return out_image.transpose(2, 0, 1).astype(np.float32)

    def _prepare_net_input(self, images, roi):
        """Converts input sequence of images into blob of data"""

        data = np.stack([self._process_image(img, roi) for img in images], axis=0)
        data = data.reshape((1,) + data.shape)
        data = np.transpose(data, (0, 2, 1, 3, 4))
        return data

    def async_infer(self, frame_buffer, person_roi, req_id):
        """Requests model inference for the specified batch of images"""

        central_roi = self._convert_to_central_roi(person_roi,
                                                   self.input_height, self.input_width,
                                                   self.img_scale)

        clip_data = self._prepare_net_input(frame_buffer, central_roi)

        super().async_infer(clip_data, req_id)

    def wait_request(self, req_id):
        """Waits for the model output"""

        result = super().wait_request(req_id)
        if result is None:
            return None
        else:
            return result[:self.num_test_classes]

    def __call__(self, frame_buffer, person_roi):
        """Runs model on the specified input"""

        central_roi = self._convert_to_central_roi(person_roi,
                                                   self.input_height, self.input_width,
                                                   self.img_scale)
        clip_data = self._prepare_net_input(frame_buffer, central_roi)

        result = self.infer(clip_data)

        return result[:self.num_test_classes]
