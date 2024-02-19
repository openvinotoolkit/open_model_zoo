# Copyright (c) 2019-2024 Intel Corporation
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
import json
import sys

from pathlib import Path

from omz_tools import _configuration, _common
from omz_tools.download_engine.downloader import Downloader


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


def download(argv):
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

    args = parser.parse_args(argv)

    reporter = Downloader.make_reporter(args.progress_format)

    with _common.telemetry_session('Model Downloader', 'downloader') as telemetry:
        args_count = sum([args.all, args.name is not None, args.list is not None, args.print_all])
        if args_count == 0:
            telemetry.send_event('md', 'downloader_selection_mode', None)
        else:
            for mode in ['all', 'list', 'name', 'print_all']:
                if getattr(args, mode):
                    telemetry.send_event('md', 'downloader_selection_mode', mode)

        models = _configuration.load_models_from_args(parser, args, _common.MODEL_ROOT)
        failed_models = set()

        if args.precisions is None:
            requested_precisions = _common.KNOWN_PRECISIONS
        else:
            requested_precisions = set(args.precisions.split(','))

        for model in models:
            precisions_to_send = requested_precisions if args.precisions else requested_precisions & model.precisions
            model_information = {
                'name': model.name,
                'framework': model.framework,
                'precisions': str(sorted(precisions_to_send)).replace(',', ';'),
            }
            telemetry.send_event('md', 'downloader_model', json.dumps(model_information))

        downloader = Downloader(requested_precisions, args.output_dir, args.cache_dir, args.num_attempts)

        failed_models = downloader.bulk_download_model(models, reporter, args.jobs, args.progress_format)

        if failed_models:
            reporter.print('FAILED:')
            for failed_model_name in failed_models:
                reporter.print(failed_model_name)
                telemetry.send_event('md', 'downloader_failed_models', failed_model_name)
            sys.exit(1)


def main():
    download(sys.argv[1:])


if __name__ == '__main__':
    main()
