"""
 Copyright (C) 2020 Intel Corporation

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

from time import perf_counter
import cv2
from helpers import put_highlighted_text


class Sum:
    def __init__(self):
        self.latency = 0.0
        self.time = 0.0
        self.frame_count = 0


class PerformanceMetrics:
    def __init__(self, time_window=1.0):    # defines the length of the timespan used for 'current fps' calculation,
        self.time_window_size = time_window # set to '1 second' by default
        self.last_moving_sum = Sum()
        self.current_moving_sum = Sum()
        self.total_sum = Sum()
        self.last_update_time = perf_counter()
    
    def update(self, last_request_start_time):
        current_time = perf_counter()

        last_request_latency = current_time - last_request_start_time
        self.current_moving_sum.latency += last_request_latency
        self.current_moving_sum.frame_count += 1
        if self.current_moving_sum.frame_count == 1 and self.total_sum.frame_count == 0:
            self.last_moving_sum.latency = last_request_latency
            self.last_moving_sum.frame_count = 1

        if current_time - self.last_update_time > self.time_window_size:
            self.current_moving_sum.time = current_time - self.last_update_time
            self.last_update_time = current_time
            self.last_moving_sum = self.current_moving_sum
            self.current_moving_sum = Sum()
            self.total_sum.latency += self.last_moving_sum.latency
            self.total_sum.time += self.last_moving_sum.time
            self.total_sum.frame_count += self.last_moving_sum.frame_count

    def get_current_fps(self):
        return self.last_moving_sum.frame_count / self.last_moving_sum.time

    def get_current_latency(self):
        return (1000 * self.last_moving_sum.latency) / self.last_moving_sum.frame_count

    def get_total_fps(self):
        return (self.total_sum.frame_count + self.current_moving_sum.frame_count) / \
               (self.total_sum.time + self.current_moving_sum.time)
        
    def get_total_latency(self):
        return (1000 * (self.total_sum.latency + self.current_moving_sum.latency)) / \
               (self.total_sum.frame_count + self.current_moving_sum.frame_count)

    def show_current_fps(self, frame, position=(15, 50), font_scale=0.75, color=(200, 10, 10), thickness=2):
        if self.last_moving_sum.frame_count > 1:
            put_highlighted_text(frame, "FPS: {:.1f}".format(self.get_current_fps()),
                                 position, cv2.FONT_HERSHEY_COMPLEX, font_scale, color, thickness)
    
    def show_current_latency(self, frame, position=(15, 20), font_scale=0.75, color=(200, 10, 10), thickness=2):
        if self.last_moving_sum.frame_count != 0:
            put_highlighted_text(frame, "Latency: {:.1f} ms".format(self.get_current_latency()),
                                 position, cv2.FONT_HERSHEY_COMPLEX, font_scale, color, thickness)
    
    def print_total_fps(self, log):
        if self.total_sum.frame_count + self.current_moving_sum.frame_count > 1:
            log.info("FPS: {:.1f}".format(self.get_total_fps()))

    def print_total_latency(self, log):
        if self.total_sum.frame_count + self.current_moving_sum.frame_count != 0:
            log.info("Latency: {:.1f} ms".format(self.get_total_latency()))
