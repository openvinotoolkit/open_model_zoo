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
from image_retrieval_demo.roi_cv_detector.detect_by_simple_dense_optical_flow import RoiDetector, \
    get_rect_tl, get_rect_br


class RoiDetectorOnVideo:
    """ This class detects moving ROI on videos. """

    def __init__(self, path):
        if not os.path.exists(path):
            raise Exception('File not found: {}'.format(path))

        self.cap = cv2.VideoCapture(path)
        self.frame_step = 5
        self.roi_detector = RoiDetector(self.frame_step)

    def __iter__(self):
        return self

    def __next__(self):
        """ Returns cropped frame (ROI) and original frame with ROI drawn as a rectangle. """

        _, frame = self.cap.read()

        if frame is None:
            raise StopIteration

        view_frame = frame.copy()

        bbox = self.roi_detector.handle_frame(frame)
        if bbox is not None:
            tl_x, tl_y = get_rect_tl(bbox)
            br_x, br_y = get_rect_br(bbox)
            frame = frame[tl_y:br_y, tl_x:br_x]
            cv2.rectangle(view_frame, (tl_x, tl_y), (br_x, br_y), (255, 0, 255), 20)
        else:
            frame = None

        return frame, view_frame
