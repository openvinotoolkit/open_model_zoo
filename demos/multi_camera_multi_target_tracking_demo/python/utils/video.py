"""
 Copyright (c) 2019-2024 Intel Corporation
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

import cv2 as cv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3] / 'common/python'))
from images_capture import open_images_capture


class MulticamCapture:
    def __init__(self, sources, loop):
        assert sources
        self.captures = []
        self.transforms = []
        self.fps = []
        for src in sources:
            capture = open_images_capture(src, loop)
            self.captures.append(capture)
            self.fps.append(capture.fps())

    def add_transform(self, t):
        self.transforms.append(t)

    def get_frames(self):
        frames = []
        for capture in self.captures:
            frame = capture.read()
            if frame is not None:
                for t in self.transforms:
                    frame = t(frame)
                frames.append(frame)

        return len(frames) == len(self.captures), frames

    def get_num_sources(self):
        return len(self.captures)

    def get_fps(self):
        return self.fps


class NormalizerCLAHE:
    def __init__(self, clip_limit=.5, tile_size=16):
        self.clahe = cv.createCLAHE(clipLimit=clip_limit,
                                    tileGridSize=(tile_size, tile_size))

    def __call__(self, frame):
        for i in range(frame.shape[2]):
            frame[:, :, i] = self.clahe.apply(frame[:, :, i])
        return frame
