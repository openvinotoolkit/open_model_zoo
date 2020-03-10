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
from multiprocessing import Process, Queue, Value

import cv2
import numpy as np

class Visualizer:
    """Class that allows to play video sources with different speed"""

    def __init__(self, trg_fps=60):
        """Constructor"""

        self._trg_time_step = 1. / float(trg_fps)
        self._last_key = Value('i', -1, lock=True)
        self._need_stop = Value('i', False, lock=False)
        self._worker_process = None
        self._tasks = dict()

    def register_window(self, name):
        """Allocates resources for the new window"""

        if self._worker_process is not None and self._worker_process.is_alive():
            raise RuntimeError('Cannot add the window for running visualizer')

        self._tasks[name] = Queue(1)

    def get_key(self):
        """Returns the value of pressed key"""

        with self._last_key.get_lock():
            out_key = self._last_key.value
            self._last_key.value = -1

        return out_key

    def get_queue(self, name):
        if name not in self._tasks:
            raise ValueError('Unknown name of queue: {}'.format(name))

        return self._tasks[name]

    def put_queue(self, frame, name):
        """Adds frame in the queue of the specified window"""

        if name not in self._tasks.keys():
            raise ValueError('Cannot show unregistered window: {}'.format(name))

        self._tasks[name].put(np.copy(frame), True)

    def start(self):
        """Starts internal threads"""

        if self._worker_process is not None and self._worker_process.is_alive():
            return

        if len(self._tasks) == 0:
            raise RuntimeError('Cannot start without registered windows')

        self._need_stop.value = False

        self._worker_process = Process(target=self._worker,
                                       args=(self._tasks, self._last_key,
                                             self._trg_time_step, self._need_stop))
        self._worker_process.daemon = True
        self._worker_process.start()

    def release(self):
        """Stops playing and releases internal storages"""

        if self._worker_process is not None:
            self._need_stop.value = True
            self._worker_process.join()

    @staticmethod
    def _worker(tasks, last_key, trg_time_step, need_stop):
        """Shows new frames in appropriate screens"""

        while not need_stop.value:
            start_time = time.perf_counter()

            for name, frame_queue in tasks.items():
                if not frame_queue.empty():
                    frame = frame_queue.get(True)

                    cv2.imshow(name, frame)

            key = cv2.waitKey(1)
            if key != -1:
                last_key.value = key

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            rest_time = trg_time_step - elapsed_time
            if rest_time > 0.0:
                time.sleep(rest_time)
