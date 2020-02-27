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

from enum import Enum
from queue import Queue


class BaseQueue:
    def __init__(self):
        self.finished = False

    def put(self, item, *args):
        if item is Signal.STOP_IMMEDIATELY:
            self.finished = True

    def task_done(self):
        pass

    def clear(self):
        pass

    def close(self):
        self.finished = True


class VoidQueue(BaseQueue):
    def put(self, item, *args):
        if item is Signal.STOP_IMMEDIATELY:
            self.close()

    def get(self):
        if self.finished:
            return Signal.STOP_IMMEDIATELY


class AsyncQueue(BaseQueue):
    def __init__(self, maxsize=0):
        super().__init__()
        self._queue = Queue(maxsize=maxsize)

    def put(self, item, block=True, timeout=None):
        if self.finished:
            return
        if item is Signal.STOP_IMMEDIATELY:
            self.close()
        else:
            self._queue.put(item, block, timeout)

    def close(self):
        self.finished = True
        with self._queue.mutex:
            self._queue.queue.clear()
            self._queue.queue.append(Signal.STOP_IMMEDIATELY)
            self._queue.unfinished_tasks = 0
            self._queue.all_tasks_done.notify()
            self._queue.not_full.notify()
            self._queue.not_empty.notify()

    def get(self, block=True, timeout=None):
        if self.finished:
            return Signal.STOP_IMMEDIATELY
        return self._queue.get(block, timeout)

    def clear(self):
        while not self._queue.empty():
            self.get()
            self.task_done()

    def task_done(self):
        if self.finished:
            return
        super().task_done()


class StubQueue(BaseQueue):
    def __init__(self):
        super().__init__()
        self.item = Signal.EMPTY

    def put(self, item, *args):
        if item is Signal.STOP_IMMEDIATELY:
            self.close()
        assert self.item is Signal.EMPTY
        self.item = item

    def get(self):
        if self.finished:
            return Signal.STOP_IMMEDIATELY
        item = self.item
        self.item = Signal.EMPTY
        assert item is not Signal.EMPTY
        return item


class Signal(Enum):
    OK = 1
    STOP = 2
    STOP_IMMEDIATELY = 3
    ERROR = 4
    EMPTY = 5


def is_stop_signal(item):
    return item is Signal.STOP or item is Signal.STOP_IMMEDIATELY
