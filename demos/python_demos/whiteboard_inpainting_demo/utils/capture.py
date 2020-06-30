"""
 Copyright (c) 2020 Intel Corporation
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
import logging as log


class VideoCapture:
    def __init__(self, source):
        try:
            cam_id = int(source)
            log.info('Connection to cam#{}'.format(cam_id))
            self.capture = cv2.VideoCapture(cam_id)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except ValueError:
            log.info('Open video file {}'.format(source))
            self.capture = cv2.VideoCapture(source)
        assert self.capture.isOpened()

    def get_frame(self):
        return self.capture.read()

    def get_source_parameters(self):
        frame_size = (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        return frame_size, fps
