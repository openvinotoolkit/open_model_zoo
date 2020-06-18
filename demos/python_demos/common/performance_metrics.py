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


class Statistic:
    def __init__(self):
        self.latency = 0.0
        self.duration = 0.0
        self.frame_count = 0


class PerformanceMetrics:
    def __init__(self, time_window=1.0):    # defines the length of the timespan used for 'current fps' calculation,
        self.time_window_size = time_window # set to '1 second' by default
        self.last_moving_statistic = Statistic()
        self.current_moving_statistic = Statistic()
        self.total_statistic = Statistic()
    
    def update(self, last_request_start_time, frame, position=(15, 30),
               font_scale=0.75, color=(200, 10, 10), thickness=2):
        current_time = perf_counter()
        if not hasattr(self, 'last_update_time'):
            self.last_update_time = current_time

        last_request_latency = current_time - last_request_start_time
        if self.current_moving_statistic.frame_count != 1 or self.total_statistic.frame_count != 0:
            self.current_moving_statistic.latency += last_request_latency
            self.current_moving_statistic.duration = current_time - self.last_update_time
            self.current_moving_statistic.frame_count += 1
        else:
            # for 1st frame
            self.last_moving_statistic.latency = last_request_latency
            self.last_moving_statistic.frame_count = 1
            self.total_statistic.latency = last_request_latency
            self.total_statistic.frame_count = 1

        if current_time - self.last_update_time > self.time_window_size:
            self.last_moving_statistic.latency = self.current_moving_statistic.latency
            self.last_moving_statistic.duration = current_time - self.last_update_time
            self.last_moving_statistic.frame_count = self.current_moving_statistic.frame_count
            self.total_statistic.latency += self.last_moving_statistic.latency
            self.total_statistic.duration += self.last_moving_statistic.duration
            self.total_statistic.frame_count += self.last_moving_statistic.frame_count
            self.current_moving_statistic.latency = 0.0
            self.current_moving_statistic.frame_count = 0

            self.last_update_time = current_time

        # Draw performance stats over frame
        current_latency, current_fps = self.get_current()
        if current_latency is not None:
            put_highlighted_text(frame, "Latency: {:.1f} ms".format(current_latency),
                                 position, cv2.FONT_HERSHEY_COMPLEX, font_scale, color, thickness)
        if current_fps is not None:
            put_highlighted_text(frame, "FPS: {:.1f}".format(current_fps),
                                 (position[0], position[1]+30), cv2.FONT_HERSHEY_COMPLEX, font_scale, color, thickness)

    def get_current(self):
        return (1000 * self.last_moving_statistic.latency / self.last_moving_statistic.frame_count
                if self.last_moving_statistic.frame_count != 0
                else None,
                self.last_moving_statistic.frame_count / self.last_moving_statistic.duration
                if self.last_moving_statistic.duration != 0.0
                else None)

    def get_total(self):
        return ((1000 * (self.total_statistic.latency + self.current_moving_statistic.latency)
                        / (self.total_statistic.frame_count + self.current_moving_statistic.frame_count))
                if self.total_statistic.frame_count + self.current_moving_statistic.frame_count != 0
                else None,
                ((self.total_statistic.frame_count + self.current_moving_statistic.frame_count)
                / (self.total_statistic.duration + self.current_moving_statistic.duration))
                if self.total_statistic.frame_count + self.current_moving_statistic.frame_count >= 2
                else None)
    
    def print_total(self):
        total_latency, total_fps = self.get_total()
        print("Latency: {:.1f} ms".format(total_latency) if total_latency is not None else "Latency: N/A")
        print("FPS: {:.1f}".format(total_fps) if total_fps is not None else "FPS: N/A")
