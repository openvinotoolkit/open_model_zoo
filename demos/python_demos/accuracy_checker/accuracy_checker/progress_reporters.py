"""
 Copyright (c) 2018 Intel Corporation

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

from tqdm import tqdm

from .dependency import ClassProvider
from .logging import print_info


class ProgressReporter(ClassProvider):
    __provider_type__ = 'progress_reporter'

    def __init__(self, dataset_size=None):
        self.finished = True
        self.dataset_size = None
        self.start_time = None
        self.prev_time = None
        self.reset(dataset_size)

    def finish(self, objects_processed=True):
        self.finished = True
        if not objects_processed:
            return
        process_time = time.time() - self.start_time
        print_info(
            '{} objects processed in {:.3f} seconds'.format(self.dataset_size, process_time))

    def reset(self, dataset_size):
        if not self.finished:
            self.finish(objects_processed=False)
        self.dataset_size = dataset_size
        self.start_time = time.time()
        self.finished = False


class PrintProgressReporter(ProgressReporter):
    __provider__ = 'print'

    def __init__(self, dataset_size, print_interval=1):
        super().__init__(dataset_size)
        self.print_interval = print_interval

    def reset(self, dataset_size):
        self.prev_time = self.start_time

    def update(self, batch_id, batch_size):
        if (batch_id + 1) % self.print_interval != 0:
            return

        t = time.time()
        batch_time = t - self.prev_time
        self.prev_time = t
        objects_processed = (batch_id + 1) * batch_size

        print_info(
            '{} / {} processed in {:.3f}s'.format(objects_processed, self.dataset_size, batch_time))


class TQDMReporter(ProgressReporter):
    __provider__ = 'bar'

    def update(self, _batch_id, batch_size):
        self.tqdm.update(batch_size)

    def finish(self, objects_processed=True):
        self.tqdm.close()
        super().finish(objects_processed)

    def reset(self, dataset_size):
        super().reset(dataset_size)
        self.tqdm = tqdm(total=self.dataset_size, unit='frames', leave=False,
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
