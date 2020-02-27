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

from collections import deque

import numpy as np


class MovingAverageMeter:
    def __init__(self, alpha):
        self.avg = None
        self.alpha = alpha

    def update(self, value):
        if self.avg is None:
            self.avg = value
            return
        self.avg = (1 - self.alpha) * self.avg + self.alpha * value

    def reset(self):
        self.avg = None


class AverageMeter:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, value):
        self.sum += value
        self.count += 1

    @property
    def avg(self):
        if self.count == 0:
            return 0
        return self.sum / self.count

    def reset(self):
        self.sum = 0
        self.count = 0


class WindowAverageMeter:
    def __init__(self, window_size=10):
        self.d = deque(maxlen=window_size)

    def update(self, value):
        self.d.append(value)

    @property
    def avg(self):
        return np.mean(self.d, axis=0)

    def reset(self):
        self.d.clear()
