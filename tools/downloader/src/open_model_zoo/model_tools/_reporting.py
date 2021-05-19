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

import json
import signal
import subprocess
import sys
import threading
import traceback

class JobContext:
    def __init__(self):
        self._interrupted = False

    def print(self, value, *, end='\n', file=sys.stdout, flush=False):
        raise NotImplementedError

    def printf(self, format, *args, file=sys.stdout, flush=False):
        self.print(format.format(*args), file=file, flush=flush)

    def subprocess(self, args, **kwargs):
        raise NotImplementedError

    def check_interrupted(self):
        if self._interrupted:
            raise RuntimeError("job interrupted")

    def interrupt(self):
        self._interrupted = True

    @staticmethod
    def _signal_message(signal_num):
        # once Python 3.8 is the minimum supported version,
        # signal.strsignal can be used here

        signals = type(signal.SIGINT)

        try:
            signal_str = f'{signals(signal_num).name} ({signal_num})'
        except ValueError:
            signal_str = f'{signal_num}'

        return f'Terminated by signal {signal_str}'

class DirectOutputContext(JobContext):
    def print(self, value, *, end='\n', file=sys.stdout, flush=False):
        print(value, end=end, file=file, flush=flush)

    def subprocess(self, args, **kwargs):
        return_code = subprocess.run(args, **kwargs).returncode

        if return_code < 0:
            print(self._signal_message(-return_code), file=sys.stderr)

        return return_code == 0


_EVENT_EMISSION_LOCK = threading.Lock()

class Reporter:
    GROUP_DECORATION = '#' * 16 + '||'
    SECTION_DECORATION = '=' * 10
    ERROR_DECORATION = '#' * 10

    def __init__(self, job_context, *,
            enable_human_output=True, enable_json_output=False, event_context={}):
        self.job_context = job_context
        self.enable_human_output = enable_human_output
        self.enable_json_output = enable_json_output
        self.event_context = event_context

    def print_group_heading(self, format, *args):
        if not self.enable_human_output: return
        self.job_context.printf('{} {} {}',
            self.GROUP_DECORATION, format.format(*args), self.GROUP_DECORATION[::-1])
        self.job_context.print('')

    def print_section_heading(self, format, *args):
        if not self.enable_human_output: return
        self.job_context.printf('{} {}', self.SECTION_DECORATION, format.format(*args), flush=True)

    def print_progress(self, format, *args):
        if not self.enable_human_output: return
        self.job_context.print(format.format(*args), end='\r' if sys.stdout.isatty() else '\n', flush=True)

    def end_progress(self):
        if not self.enable_human_output: return
        if sys.stdout.isatty():
            self.job_context.print('')

    def print(self, format='', *args, flush=False):
        if not self.enable_human_output: return
        self.job_context.printf(format, *args, flush=flush)

    def log_warning(self, format, *args, exc_info=False):
        if exc_info:
            self.job_context.print(traceback.format_exc(), file=sys.stderr, end='')
        self.job_context.printf("{} Warning: {}", self.ERROR_DECORATION, format.format(*args), file=sys.stderr)

    def log_error(self, format, *args, exc_info=False):
        if exc_info:
            self.job_context.print(traceback.format_exc(), file=sys.stderr, end='')
        self.job_context.printf("{} Error: {}", self.ERROR_DECORATION, format.format(*args), file=sys.stderr)

    def log_details(self, format, *args):
        print(self.ERROR_DECORATION, '    ', format.format(*args), file=sys.stderr)

    def emit_event(self, type, **kwargs):
        if not self.enable_json_output: return

        # We don't print machine-readable output through the job context, because
        # we don't want it to be serialized. If we serialize it, then the consumer
        # will lose information about the order of events, and we don't want that to happen.
        # Instead, we emit events directly to stdout, but use a lock to ensure that
        # JSON texts don't get interleaved.
        with _EVENT_EMISSION_LOCK:
            json.dump({'$type': type, **self.event_context, **kwargs}, sys.stdout, indent=None)
            print()

    def with_event_context(self, **kwargs):
        return Reporter(
            self.job_context,
            enable_human_output=self.enable_human_output,
            enable_json_output=self.enable_json_output,
            event_context={**self.event_context, **kwargs},
        )
