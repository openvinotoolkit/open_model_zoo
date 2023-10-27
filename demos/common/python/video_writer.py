"""
 Copyright (C) 2020-2023 Intel Corporation

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

from cv2 import VideoWriter
import logging as log

DEFAULT_LIMIT_WARN_THROTTLE_INTERVAL_FRAMES = 500

def on_limit_reached_default(frame, number, limit):
    if (number - limit) % DEFAULT_LIMIT_WARN_THROTTLE_INTERVAL_FRAMES == 1:
        log.warning("VideoWriter will skip writing next frame due to frame limit applied: {0}.\nIf you want to turn off this limitation please set `-limit 0`".format(limit))

class LazyVideoWriter(VideoWriter):
    def __init__(self):
        VideoWriter.__init__(self)

    def open(self, output, fourcc, fps, output_limit, output_resolution, on_limit_callback = on_limit_reached_default):
        super().open(output, fourcc, fps, output_resolution)

        self.output_limit = output_limit
        self.frames_processed = 0
        self.on_limit_callback = on_limit_callback

    def write(self, frame):
        if super().isOpened():
            if self.output_limit <= 0 or self.frames_processed <= self.output_limit:
                super().write(frame)
            else:
                self.on_limit_callback(frame, self.frames_processed, self.output_limit)

        self.frames_processed += 1


