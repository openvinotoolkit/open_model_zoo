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

from collections import deque
from time import perf_counter

class Measurement:
    def __init__(self, latency, time_point):
        self.latency = latency
        self.time_point = time_point

class PerformanceMetrics:
    def __init__(self, time_window=1):      # defines the length of the timespan used for 'current fps' calculation,
        self.time_window_size = time_window # set to '1 second' by default
        self.measurements = deque()
        self.reinitialize()
    
    def recalculate(self, last_request_start_time):
        current_time = perf_counter()
        while self.measurements:
            first_in_window = self.measurements[0]
            if current_time - first_in_window.time_point > self.time_window_size:
                self.latency_sum -= first_in_window.latency
                self.measurements.popleft()
            else:
                break

        last_request_latency = current_time - last_request_start_time
        self.measurements.append(Measurement(last_request_latency, current_time))
        self.latency_sum += last_request_latency
        self.latency_total_sum += last_request_latency
        self.latency = (1000 * self.latency_sum) / len(self.measurements)

        spf_sum = self.measurements[-1].time_point - self.measurements[0].time_point
        if spf_sum > 0:
            self.fps = len(self.measurements) / spf_sum

        self.num_frames_processed += 1

    def get_total_fps(self):
        return self.num_frames_processed / (self.stop_time - self.start_time)
        
    def get_total_latency(self):
        return (1000 * self.latency_total_sum) / self.num_frames_processed

    def stop(self):
        self.stop_time = perf_counter()

    def reinitialize(self):
        self.measurements.clear()
        self.fps = 0
        self.latency = 0
        self.latency_sum = 0
        self.start_time = perf_counter()
        self.num_frames_processed = 0
        self.latency_total_sum = 0

    def has_started(self):
        return self.num_frames_processed > 0
