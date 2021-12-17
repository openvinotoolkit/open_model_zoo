"""
 Copyright (c) 2018-2021 Intel Corporation

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

import argparse
import contextlib
import json
import requests
import sys
import threading

from pathlib import Path

from openvino.model_zoo import (
    _configuration, _common, _concurrency, _reporting,
)
from openvino.model_zoo.download_engine.downloader import Downloader


class DownloaderArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def positive_int_arg(value_str):
    try:
        value = int(value_str)
        if value > 0: return value
    except ValueError:
        pass

    raise argparse.ArgumentTypeError('must be a positive integer (got {!r})'.format(value_str))


# There is no evidence that the requests.Session class is thread-safe,
# so for safety, we use one Session per thread. This class ensures that
# each thread gets its own Session.
class ThreadSessionFactory:
    def __init__(self, exit_stack):
        self._lock = threading.Lock()
        self._thread_local = threading.local()
        self._exit_stack = exit_stack

    def __call__(self):
        try:
            session = self._thread_local.session
        except AttributeError:
            with self._lock: # ExitStack might not be thread-safe either
                session = self._exit_stack.enter_context(requests.Session())
            self._thread_local.session = session
        return session


def main():
    parser = DownloaderArgumentParser()
    parser.add_argument('--name', metavar='PAT[,PAT...]',
        help='download only models whose names match at least one of the specified patterns')
    parser.add_argument('--list', type=Path, metavar='FILE.LST',
        help='download only models whose names match at least one of the patterns in the specified file')
    parser.add_argument('--all', action='store_true', help='download all available models')
    parser.add_argument('--print_all', action='store_true', help='print all available models')
    parser.add_argument('--precisions', metavar='PREC[,PREC...]',
                        help='download only models with the specified precisions (actual for DLDT networks); specify one or more of: '
                             + ','.join(_common.KNOWN_PRECISIONS))
    parser.add_argument('-o', '--output_dir', type=Path, metavar='DIR',
        default=Path.cwd(), help='path where to save models')
    parser.add_argument('--cache_dir', type=Path, metavar='DIR',
        help='directory to use as a cache for downloaded files')
    parser.add_argument('--num_attempts', type=positive_int_arg, metavar='N', default=1,
        help='attempt each download up to N times')
    parser.add_argument('--progress_format', choices=('text', 'json'), default='text',
        help='which format to use for progress reporting')
    # unlike Model Converter, -jauto is not supported here, because CPU count has no
    # relation to the optimal number of concurrent downloads
    parser.add_argument('-j', '--jobs', type=positive_int_arg, metavar='N', default=1,
        help='how many downloads to perform concurrently')

    args = parser.parse_args()

    def make_reporter(context):
        return _reporting.Reporter(context,
            enable_human_output=args.progress_format == 'text',
            enable_json_output=args.progress_format == 'json')

    reporter = make_reporter(_reporting.DirectOutputContext())

    with _common.telemetry_session('Model Downloader', 'downloader') as telemetry:
        models = _configuration.load_models_from_args(parser, args, _common.MODEL_ROOT)

        for mode in ['all', 'list', 'name']:
            if getattr(args, mode):
                telemetry.send_event('md', 'downloader_selection_mode', mode)

        if args.precisions is None:
            requested_precisions = _common.KNOWN_PRECISIONS
        else:
            requested_precisions = set(args.precisions.split(','))

        for model in models:
            precisions_to_send = requested_precisions if args.precisions else requested_precisions & model.precisions
            model_information = {
                'name': model.name,
                'framework': model.framework,
                'precisions': str(precisions_to_send).replace(',', ';'),
            }
            telemetry.send_event('md', 'downloader_model', json.dumps(model_information))

        failed_models = set()

        unknown_precisions = requested_precisions - _common.KNOWN_PRECISIONS
        if unknown_precisions:
            sys.exit('Unknown precisions specified: {}.'.format(', '.join(sorted(unknown_precisions))))

        downloader = Downloader(args.output_dir, args.cache_dir, args.num_attempts)

        def download_model(model, reporter, session):
            if model.model_stages:
                results = []
                for model_stage in model.model_stages:
                    results.append(downloader.download_model(
                        reporter, session, requested_precisions, model_stage, _common.KNOWN_PRECISIONS))
                return sum(results) == len(model.model_stages)
            else:
                return downloader.download_model(
                    reporter, session, requested_precisions, model, _common.KNOWN_PRECISIONS)

        with contextlib.ExitStack() as exit_stack:
            session_factory = ThreadSessionFactory(exit_stack)
            if args.jobs == 1:
                results = [download_model(model, reporter, session_factory) for model in models]
            else:
                results = _concurrency.run_in_parallel(args.jobs,
                    lambda context, model: download_model(model, make_reporter(context), session_factory),
                    models)

        failed_models = {model.name for model, successful in zip(models, results) if not successful}

        if failed_models:
            reporter.print('FAILED:')
            for failed_model_name in failed_models:
                reporter.print(failed_model_name)
                telemetry.send_event('md', 'downloader_failed_models', failed_model_name)
            sys.exit(1)

if __name__ == '__main__':
    main()
