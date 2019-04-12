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
