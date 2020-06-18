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

        try:
            sources = [int(src) for src in sources]
            mode = 'cam'
        except ValueError:
            mode = 'video'

        if mode == 'cam':
            for id in sources:
                log.info('Connection  cam {}'.format(id))
                cap = cv.VideoCapture(id)
                cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv.CAP_PROP_FPS, 30)
                cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
                assert cap.isOpened()
                self.captures.append(cap)
        else:
            for video_path in sources:
                log.info('Opening file {}'.format(video_path))
                cap = cv.VideoCapture(video_path)
                assert cap.isOpened()
                self.captures.append(cap)

    def add_transform(self, t):
        self.transforms.append(t)

    def get_frames(self):
        frames = []
        for capture in self.captures:
            has_frame, frame = capture.read()
            if has_frame:
                for t in self.transforms:
                    frame = t(frame)
                frames.append(frame)

        return len(frames) == len(self.captures), frames

    def get_num_sources(self):
        return len(self.captures)

    def get_source_parameters(self):
        frame_size = []
        fps = []
        for cap in self.captures:
            frame_size.append((int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                               int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
            fps.append(int(cap.get(cv.CAP_PROP_FPS)))
        return frame_size, fps


class NormalizerCLAHE:
    def __init__(self, clip_limit=.5, tile_size=16):
        self.clahe = cv.createCLAHE(clipLimit=clip_limit,
                                    tileGridSize=(tile_size, tile_size))

    def __call__(self, frame):
        for i in range(frame.shape[2]):
            frame[:, :, i] = self.clahe.apply(frame[:, :, i])
        return frame
