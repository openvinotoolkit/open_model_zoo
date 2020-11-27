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

import logging as log
import cv2 as cv


class MulticamCapture:
    def __init__(self, sources):
        assert sources
        self.captures = []
        self.transforms = []
        self.frame_size = []
        self.fps = []
        for src in sources:
            cap = open_images_capture(src, False)
            fps = cap.fps()
            self.captures.append(cap)
            self.fps.append(fps)

    def add_transform(self, t):
        self.transforms.append(t)

    def get_frames(self):
        frames = []
        for capture in self.captures:
            frame = capture.read()
            if frame is None:
                raise RuntimeError("Can't read an image from the input")
            frame_size = frame.shape
            self.frame_size.append((frame_size[1], frame_size[0]))
            for t in self.transforms:
                frame = t(frame)
            frames.append(frame)

        return len(frames) == len(self.captures), frames

    def get_num_sources(self):
        return len(self.captures)

    def get_source_parameters(self):
        return self.frame_size, self.fps


class NormalizerCLAHE:
    def __init__(self, clip_limit=.5, tile_size=16):
        self.clahe = cv.createCLAHE(clipLimit=clip_limit,
                                    tileGridSize=(tile_size, tile_size))

    def __call__(self, frame):
        for i in range(frame.shape[2]):
            frame[:, :, i] = self.clahe.apply(frame[:, :, i])
        return frame
