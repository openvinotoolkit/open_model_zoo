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
from os import path as osp
import sys


ESC = 27
SPACE = 32
ENTER = 13

class MouseClick:
    def __init__(self):
        self.points = {}
        self.crop_available = False

    def get_points(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points[0] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.points[1] = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points = {}
        self.crop_available = True if len(self.points) == 2 else False


def set_log_config():
    log.basicConfig(stream=sys.stdout, format='%(levelname)s: %(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=log.DEBUG)


def check_pressed_keys(key):
    if key == SPACE:  # Pause
        while True:
            key = cv2.waitKey(0)
            if key == ESC or key == SPACE or key == ENTER:  # enter: resume, space: next frame, esc: exit
                break
    else:
        key = cv2.waitKey(1)
    return key
