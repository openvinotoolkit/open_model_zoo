# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
import queue
import subprocess
import sys

from open_model_zoo.model_tools import _reporting


class _QueuedOutputContext(_reporting.JobContext):
    def __init__(self, output_queue):
        super().__init__()
        self._output_queue = output_queue

    def print(self, value, *, end='\n', file=sys.stdout, flush=False):
        self._output_queue.put((file, value + end))

    def subprocess(self, args, **kwargs):
        with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, **kwargs) as p:
            for line in p.stdout:
                self._output_queue.put((sys.stdout, line))
            return_code = p.wait()

        if return_code < 0:
            self._output_queue.put((sys.stderr, self._signal_message(-return_code)))

        return return_code == 0


class _JobWithQueuedOutput():
    def __init__(self, context, output_queue, future):
        self._context = context
        self._output_queue = output_queue
        self._future = future
        self._future.add_done_callback(lambda future: self._output_queue.put(None))

    def complete(self):
        for file, fragment in iter(self._output_queue.get, None):
            print(fragment, end='', file=file, flush=True) # for simplicity, flush every fragment

        return self._future.result()

    def cancel(self):
        self._context.interrupt()
        self._future.cancel()


def run_in_parallel(num_jobs, f, work_items):
    with concurrent.futures.ThreadPoolExecutor(num_jobs) as executor:
        def start(work_item):
            output_queue = queue.Queue()
            context = _QueuedOutputContext(output_queue)
            return _JobWithQueuedOutput(
                context, output_queue, executor.submit(f, context, work_item))

        jobs = list(map(start, work_items))

        try:
            return [job.complete() for job in jobs]
        except BaseException:
            for job in jobs: job.cancel()
            raise
