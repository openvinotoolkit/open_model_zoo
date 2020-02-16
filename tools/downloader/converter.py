#!/usr/bin/env python3

# Copyright (c) 2019 Intel Corporation
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

import argparse
import concurrent.futures
import os
import platform
import queue
import re
import shlex
import string
import subprocess
import sys
import threading

from pathlib import Path

import common

class JobContext:
    def printf(self, format, *args, flush=False):
        raise NotImplementedError

    def subprocess(self, args):
        raise NotImplementedError


class DirectOutputContext(JobContext):
    def printf(self, format, *args, flush=False):
        print(format.format(*args), flush=flush)

    def subprocess(self, args):
        return subprocess.run(args).returncode == 0


class QueuedOutputContext(JobContext):
    def __init__(self, output_queue):
        self._output_queue = output_queue

    def printf(self, format, *args, flush=False):
        self._output_queue.put(format.format(*args) + '\n')

    def subprocess(self, args):
        with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True) as p:
            for line in p.stdout:
                self._output_queue.put(line)
            return p.wait() == 0


class JobWithQueuedOutput():
    def __init__(self, output_queue, future):
        self._output_queue = output_queue
        self._future = future
        self._future.add_done_callback(lambda future: self._output_queue.put(None))

    def complete(self):
        for fragment in iter(self._output_queue.get, None):
            print(fragment, end='', flush=True) # for simplicity, flush every fragment

        return self._future.result()

    def cancel(self):
        self._future.cancel()


def quote_windows(arg):
    if not arg: return '""'
    if not re.search(r'\s|"', arg): return arg
    # On Windows, only backslashes that precede a quote or the end of the argument must be escaped.
    return '"' + re.sub(r'(\\+)$', r'\1\1', re.sub(r'(\\*)"', r'\1\1\\"', arg)) + '"'

if platform.system() == 'Windows':
    quote_arg = quote_windows
else:
    quote_arg = shlex.quote

def convert_to_onnx(context, model, output_dir, args):
    context.printf('========= {}Converting {} to ONNX',
                   '(DRY RUN) ' if args.dry_run else '', model.name)

    conversion_to_onnx_args = [string.Template(arg).substitute(conv_dir=output_dir / model.subdirectory,
                                                               dl_dir=args.download_dir / model.subdirectory)
                               for arg in model.conversion_to_onnx_args]
    cmd = [str(args.python), str(Path(__file__).absolute().parent / model.converter_to_onnx), *conversion_to_onnx_args]

    context.printf('Conversion to ONNX command: {}', ' '.join(map(quote_arg, cmd)))
    context.printf('')

    success = True if args.dry_run else context.subprocess(cmd)
    context.printf('')

    return success

def num_jobs_arg(value_str):
    if value_str == 'auto':
        return os.cpu_count() or 1

    try:
        value = int(value_str)
        if value > 0: return value
    except ValueError:
        pass

    raise argparse.ArgumentTypeError('must be a positive integer or "auto" (got {!r})'.format(value_str))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--download_dir', type=Path, metavar='DIR',
        default=Path.cwd(), help='root of the directory tree with downloaded model files')
    parser.add_argument('-o', '--output_dir', type=Path, metavar='DIR',
        help='root of the directory tree to place converted files into')
    parser.add_argument('--name', metavar='PAT[,PAT...]',
        help='convert only models whose names match at least one of the specified patterns')
    parser.add_argument('--list', type=Path, metavar='FILE.LST',
        help='convert only models whose names match at least one of the patterns in the specified file')
    parser.add_argument('--all', action='store_true', help='convert all available models')
    parser.add_argument('--print_all', action='store_true', help='print all available models')
    parser.add_argument('--precisions', metavar='PREC[,PREC...]',
        help='run only conversions that produce models with the specified precisions')
    parser.add_argument('-p', '--python', type=Path, metavar='PYTHON', default=sys.executable,
        help='Python executable to run Model Optimizer with')
    parser.add_argument('--mo', type=Path, metavar='MO.PY',
        help='Model Optimizer entry point script')
    parser.add_argument('--add-mo-arg', dest='extra_mo_args', metavar='ARG', action='append',
        help='Extra argument to pass to Model Optimizer')
    parser.add_argument('--dry-run', action='store_true',
        help='Print the conversion commands without running them')
    parser.add_argument('-j', '--jobs', type=num_jobs_arg, default=1,
        help='number of conversions to run concurrently')
    args = parser.parse_args()

    mo_path = args.mo
    if mo_path is None:
        try:
            mo_path = Path(os.environ['INTEL_OPENVINO_DIR']) / 'deployment_tools/model_optimizer/mo.py'
        except KeyError:
            sys.exit('Unable to locate Model Optimizer. '
                + 'Use --mo or run setupvars.sh/setupvars.bat from the OpenVINO toolkit.')

    extra_mo_args = args.extra_mo_args or []

    if args.precisions is None:
        requested_precisions = common.KNOWN_PRECISIONS
    else:
        requested_precisions = set(args.precisions.split(','))
        unknown_precisions = requested_precisions - common.KNOWN_PRECISIONS
        if unknown_precisions:
            sys.exit('Unknown precisions specified: {}.'.format(', '.join(sorted(unknown_precisions))))

    models = common.load_models_from_args(parser, args)

    output_dir = args.download_dir if args.output_dir is None else args.output_dir

    def convert(context, model):
        if model.mo_args is None:
            context.printf('========= Skipping {} (no conversions defined)', model.name)
            context.printf('')
            return True

        model_precisions = requested_precisions & model.precisions
        if not model_precisions:
            context.printf('========= Skipping {} (all conversions skipped)', model.name)
            context.printf('')
            return True

        model_format = model.framework

        if model.conversion_to_onnx_args:
            if not convert_to_onnx(context, model, output_dir, args):
                return False
            model_format = 'onnx'

        expanded_mo_args = [
            string.Template(arg).substitute(dl_dir=args.download_dir / model.subdirectory,
                                            mo_dir=mo_path.parent,
                                            conv_dir=output_dir / model.subdirectory)
            for arg in model.mo_args]

        for model_precision in sorted(model_precisions):
            mo_cmd = [str(args.python), '--', str(mo_path),
                '--framework={}'.format(model_format),
                '--data_type={}'.format(model_precision),
                '--output_dir={}'.format(output_dir / model.subdirectory / model_precision),
                '--model_name={}'.format(model.name),
                *expanded_mo_args, *extra_mo_args]

            context.printf('========= {}Converting {} to IR ({})',
                '(DRY RUN) ' if args.dry_run else '', model.name, model_precision)

            context.printf('Conversion command: {}', ' '.join(map(quote_arg, mo_cmd)))

            if not args.dry_run:
                context.printf('', flush=True)

                if not context.subprocess(mo_cmd):
                    return False

            context.printf('')

        return True

    if args.jobs == 1 or args.dry_run:
        context = DirectOutputContext()
        results = [convert(context, model) for model in models]
    else:
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as executor:
            def start(model):
                output_queue = queue.Queue()
                return JobWithQueuedOutput(
                    output_queue,
                    executor.submit(convert, QueuedOutputContext(output_queue), model))

            jobs = list(map(start, models))

            try:
                results = [job.complete() for job in jobs]
            except:
                for job in jobs: job.cancel()
                raise

    failed_models = [model.name for model, successful in zip(models, results) if not successful]

    if failed_models:
        print('FAILED:')
        print(*sorted(failed_models), sep='\n')
        sys.exit(1)

if __name__ == '__main__':
    main()
