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

import logging as log
from time import perf_counter

import cv2


def put_highlighted_text(frame, message, position, font_face, font_scale, color, thickness):
    cv2.putText(frame, message, position, font_face, font_scale, (255, 255, 255), thickness + 1) # white border
    cv2.putText(frame, message, position, font_face, font_scale, color, thickness)


class Statistic:
    def __init__(self):
        self.latency = 0.0
        self.period = 0.0
        self.frame_count = 0

    def combine(self, other):
        self.latency += other.latency
        self.period += other.period
        self.frame_count += other.frame_count


class PerformanceMetrics:
    def __init__(self, time_window=1.0):
        # 'time_window' defines the length of the timespan over which the 'current fps' value is calculated
        self.time_window_size = time_window
        self.last_moving_statistic = Statistic()
        self.current_moving_statistic = Statistic()
        self.total_statistic = Statistic()
        self.last_update_time = None

    def update(self, last_request_start_time, frame=None):
        current_time = perf_counter()

        if self.last_update_time is None:
            self.last_update_time = last_request_start_time

        self.current_moving_statistic.latency += current_time - last_request_start_time
        self.current_moving_statistic.period = current_time - self.last_update_time
        self.current_moving_statistic.frame_count += 1

        if current_time - self.last_update_time > self.time_window_size:
            self.last_moving_statistic = self.current_moving_statistic
            self.total_statistic.combine(self.last_moving_statistic)
            self.current_moving_statistic = Statistic()
            self.last_update_time = current_time

        if frame is not None:
            self.paint_metrics(frame)

    def paint_metrics(self, frame, position=(15, 30), font_scale=0.75, color=(200, 10, 10), thickness=2):
        # Draw performance stats over frame
        current_latency, current_fps = self.get_last()
        if current_latency is not None:
            put_highlighted_text(frame, "Latency: {:.1f} ms".format(current_latency * 1e3),
                                 position, cv2.FONT_HERSHEY_COMPLEX, font_scale, color, thickness)
        if current_fps is not None:
            put_highlighted_text(frame, "FPS: {:.1f}".format(current_fps),
                                 (position[0], position[1]+30), cv2.FONT_HERSHEY_COMPLEX, font_scale, color, thickness)

    def get_last(self):
        return (self.last_moving_statistic.latency / self.last_moving_statistic.frame_count
                if self.last_moving_statistic.frame_count != 0
                else None,
                self.last_moving_statistic.frame_count / self.last_moving_statistic.period
                if self.last_moving_statistic.period != 0.0
                else None)

    def get_total(self):
        frame_count = self.total_statistic.frame_count + self.current_moving_statistic.frame_count
        return (((self.total_statistic.latency + self.current_moving_statistic.latency) / frame_count)
                if frame_count != 0
                else None,
                (frame_count / (self.total_statistic.period + self.current_moving_statistic.period))
                if frame_count != 0
                else None)

    def get_latency(self):
        return self.get_total()[0] * 1e3

    def log_total(self):
        total_latency, total_fps = self.get_total()
        log.info('Metrics report:')
        log.info("\tLatency: {:.1f} ms".format(total_latency * 1e3) if total_latency is not None else "\tLatency: N/A")
        log.info("\tFPS: {:.1f}".format(total_fps) if total_fps is not None else "\tFPS: N/A")
