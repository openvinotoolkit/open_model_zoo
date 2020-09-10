"""
Copyright (c) 2018-2020 Intel Corporation

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
import warnings

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .dependency import ClassProvider
from .logging import print_info


class ProgressReporter(ClassProvider):
    __provider_type__ = 'progress_reporter'

    def __init__(self, dataset_size=None, **kwargs):
        self.finished = True
        self.dataset_size = None
        self.start_time = None
        self.prev_time = None
        if dataset_size is not None:
            self.reset(dataset_size)
        self.current = 0

    def finish(self, objects_processed=True):
        self.finished = True
        if not objects_processed:
            return

        process_time = time.time() - self.start_time
        print_info('{} objects processed in {:.3f} seconds'.format(self.dataset_size, process_time))

    @property
    def progress(self):
        return (self.current / self.dataset_size) * 100 if self.dataset_size else 0

    def reset(self, dataset_size):
        if not self.finished:
            self.finish(objects_processed=False)
        self.current = 0

        self.dataset_size = dataset_size
        self.start_time = time.time()
        self.finished = False


class PrintProgressReporter(ProgressReporter):
    __provider__ = 'print'

    def __init__(self, dataset_size=None, print_interval=1000):
        super().__init__(dataset_size)
        self.print_interval = print_interval

    def reset(self, dataset_size, objects='dataset'):
        self.dataset_size = dataset_size
        print_info('Total {objects} size: {size}'.format(objects=objects, size=dataset_size))
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.current = 0

    def update(self, batch_id, batch_size):
        self.current += batch_size
        if (batch_id + 1) % self.print_interval != 0:
            return

        now = time.time()
        batch_time = now - self.prev_time
        self.prev_time = now

        print_info('{} / {} processed in {:.3f}s'.format((batch_id + 1) * batch_size, self.dataset_size, batch_time))


class TQDMReporter(ProgressReporter):
    __provider__ = 'bar'

    def __init__(self, dataset_size=None, **kwargs):
        super().__init__(dataset_size)
        if tqdm is None:
            warnings.warn('tqdm is not available, progress switched to print')
            self.tqdm_reporter = None
            self.progress_printer = PrintProgressReporter(dataset_size, print_interval=1)
        else:
            self.tqdm_reporter = tqdm

    def update(self, batch_id, batch_size):
        self.current += batch_size
        if self.tqdm_reporter is not None:
            self.tqdm_reporter.update(batch_size)
        else:
            self.progress_printer.update(batch_id, batch_size)

    def finish(self, objects_processed=True):
        if self.tqdm_reporter is not None:
            self.tqdm_reporter.close() #pylint: disable=E1120
            super().finish(objects_processed)
        else:
            self.progress_printer.finish(objects_processed)

    def reset(self, dataset_size):
        super().reset(dataset_size)
        if self.tqdm_reporter is not None:
            self.tqdm_reporter = tqdm(
                total=self.dataset_size, unit='frames', leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        else:
            self.progress_printer.reset(dataset_size)
