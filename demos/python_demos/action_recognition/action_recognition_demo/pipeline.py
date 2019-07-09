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
from collections import OrderedDict
from itertools import chain, cycle
from threading import Thread

from .queue import AsyncQueue, Signal, StubQueue, VoidQueue, is_stop_signal
from .timer import TimerGroup, IncrementalTimer


class PipelineStep:
    def __init__(self):
        self.input_queue = None
        self.output_queue = VoidQueue()
        self.working = False
        self.timers = TimerGroup()
        self.total_time = IncrementalTimer()
        self.own_time = IncrementalTimer()

        self._start_t = None
        self._thread = None

    def process(self, item):
        raise NotImplementedError

    def end(self):
        pass

    def setup(self):
        pass

    def start(self):
        if self.input_queue is None or self.output_queue is None:
            raise Exception("No input or output queue")

        if self._thread is not None:
            raise Exception("Thread is already running")
        self._thread = Thread(target=self._run)
        self._thread.start()
        self.working = True

    def join(self):
        print("finishing {}".format(self))
        self.input_queue.put(Signal.STOP)
        self._thread.join()
        self._thread = None
        self.working = False

    def _run(self):
        self._start_t = time.time()
        self.setup()

        self.total_time = IncrementalTimer()
        self.own_time = IncrementalTimer()

        while True:
            self.total_time.tick()
            item = self.input_queue.get()
            # print("{} get".format(self))

            if self._check_output(item):
                break

            self.own_time.tick()
            output = self.process(item)
            self.own_time.tock()

            if self._check_output(output):
                break

            self.total_time.tock()
            self.input_queue.task_done()
            self.output_queue.put(output)

        self.input_queue.close()
        self.end()
        self.working = False

    def _check_output(self, item):
        if is_stop_signal(item):
            self.output_queue.put(item)
            return True
        return False


class AsyncPipeline:
    def __init__(self):
        self.steps = OrderedDict()
        self.sync_steps = OrderedDict()
        self.async_step = []

        self._void_queue = VoidQueue()
        self._last_step = None
        self._last_parallel = False

    def add_step(self, name, new_pipeline_step, max_size=100, parallel=True):
        new_pipeline_step.output_queue = self._void_queue
        if self._last_step:
            if parallel or self._last_parallel:
                queue = AsyncQueue(maxsize=max_size)
            else:
                queue = StubQueue()

            self._last_step.output_queue = queue
            new_pipeline_step.input_queue = queue
        else:
            new_pipeline_step.input_queue = self._void_queue

        if parallel:
            self.steps[name] = new_pipeline_step
        else:
            self.sync_steps[name] = new_pipeline_step
        self._last_step = new_pipeline_step
        self._last_parallel = parallel

    def run(self):
        for step in self.steps.values():
            if not step.working:
                step.start()
        self._run_sync_steps()

    def close(self):
        for step in self.steps.values():
            step.input_queue.put(Signal.STOP_IMMEDIATELY)
        for step in self.steps.values():
            step.join()

    def print_statistics(self):
        for name, step in chain(self.sync_steps.items(), self.steps.items(), ):
            print("{} total: {}".format(name, step.total_time))
            print("{}   own: {}".format(name, step.own_time))

    def _run_sync_steps(self):
        """Run steps in main thread"""
        if not self.sync_steps:
            while not self._void_queue.finished:
                pass
            return

        for step in self.sync_steps.values():
            step.working = True
            step.setup()

        for step in cycle(self.sync_steps.values()):
            step.total_time.tick()
            item = step.input_queue.get()

            if is_stop_signal(item):
                step.input_queue.close()
                step.output_queue.put(item)
                break

            step.own_time.tick()
            output = step.process(item)
            step.own_time.tock()

            if is_stop_signal(output):
                step.input_queue.close()
                step.output_queue.put(output)
                break

            step.total_time.tock()
            step.output_queue.put(output)

        for step in self.sync_steps.values():
            step.working = False
            step.end()
