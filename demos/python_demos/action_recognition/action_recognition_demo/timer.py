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

import time
from collections import defaultdict
from contextlib import contextmanager
from math import sqrt


class IncrementalTimer:
    def __init__(self):
        self.start_t = None
        self.total_ms = 0
        self.last = 0
        self._sum_sq = 0
        self._times = 0

    def tick(self):
        self.start_t = time.perf_counter()

    def tock(self):
        now = time.perf_counter()
        elapsed_ms = (now - self.start_t) * 1000.

        self.total_ms += elapsed_ms
        self._sum_sq += elapsed_ms ** 2
        self._times += 1
        self.last = elapsed_ms

    @property
    def fps(self):
        return 1000 / self.avg

    @property
    def avg(self):
        """Returns average time in ms"""
        return self.total_ms / self._times

    @property
    def std(self):
        return sqrt((self._sum_sq / self._times) - self.avg ** 2)

    @contextmanager
    def time_section(self):
        self.tick()
        yield
        self.tock()

    def __repr__(self):
        return "{:.2f}ms (±{:.2f}) {:.2f}fps".format(self.avg, self.std, self.fps)


class TimerGroup:
    def __init__(self):
        self.timers = defaultdict(IncrementalTimer)

    def tick(self, timer):
        self.timers[timer].tick()

    def tock(self, timer):
        self.timers[timer].tock()

    @contextmanager
    def time_section(self, timer):
        self.tick(timer)
        yield
        self.tock(timer)

    def print_statistics(self):
        for name, timer in self.timers.items():
            print("{}: {}".format(name, timer))
