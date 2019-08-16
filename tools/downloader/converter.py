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
import re
import shlex
import string
import subprocess
import sys
import threading

from pathlib import Path

import common

def quote_windows(arg):
    if not arg: return '""'
    if not re.search(r'\s|"', arg): return arg
    # On Windows, only backslashes that precede a quote or the end of the argument must be escaped.
    return '"' + re.sub(r'(\\+)$', r'\1\1', re.sub(r'(\\*)"', r'\1\1\\"', arg)) + '"'

if platform.system() == 'Windows':
    quote_arg = quote_windows
else:
    quote_arg = shlex.quote

def prefixed_printf(prefix, format, *args, **kwargs):
    if prefix is None:
        print(format.format(*args), **kwargs)
    else:
        print(prefix + ': ' + format.format(*args), **kwargs)

def prefixed_subprocess(prefix, args):
    if prefix is None:
        return subprocess.run(args).returncode == 0

    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True) as p:
        for line in p.stdout:
            sys.stdout.write(prefix + ': ' + line)
        return p.wait() == 0

def convert_to_onnx(topology, output_dir, args, stdout_prefix):
    pytorch_converter = Path(__file__).absolute().parent / 'pytorch_to_onnx.py'
    prefixed_printf(stdout_prefix, '========= {}Converting {} to ONNX',
        '(DRY RUN) ' if args.dry_run else '', topology.name)

    pytorch_to_onnx_args = [string.Template(arg).substitute(conv_dir=output_dir / topology.subdirectory,
                                                            dl_dir=args.download_dir / topology.subdirectory)
                            for arg in topology.pytorch_to_onnx_args]
    cmd = [str(args.python), str(pytorch_converter), *pytorch_to_onnx_args]
    prefixed_printf(stdout_prefix, 'Conversion to ONNX command: {}', ' '.join(map(quote_arg, cmd)))

    return True if args.dry_run else prefixed_subprocess(stdout_prefix, cmd)

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
    parser.add_argument('-c', '--config', type=Path, metavar='CONFIG.YML',
        help='topology configuration file (deprecated)')
    parser.add_argument('-d', '--download_dir', type=Path, metavar='DIR',
        default=Path.cwd(), help='root of the directory tree with downloaded topology files')
    parser.add_argument('-o', '--output_dir', type=Path, metavar='DIR',
        help='root of the directory tree to place converted files into')
    parser.add_argument('--name', metavar='PAT[,PAT...]',
        help='convert only topologies whose names match at least one of the specified patterns')
    parser.add_argument('--list', type=Path, metavar='FILE.LST',
        help='convert only topologies whose names match at least one of the patterns in the specified file')
    parser.add_argument('--all', action='store_true', help='convert all topologies from the configuration file')
    parser.add_argument('--print_all', action='store_true', help='print all available topologies')
    parser.add_argument('--precisions', metavar='PREC[,PREC...]',
        help='run only conversions that produce models with the specified precisions')
    parser.add_argument('-p', '--python', type=Path, metavar='PYTHON', default=sys.executable,
        help='Python executable to run Model Optimizer with')
    parser.add_argument('--mo', type=Path, metavar='MO.PY',
        help='Model Optimizer entry point script')
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

    if args.precisions is None:
        requested_precisions = common.KNOWN_PRECISIONS
    else:
        requested_precisions = set(args.precisions.split(','))
        unknown_precisions = requested_precisions - common.KNOWN_PRECISIONS
        if unknown_precisions:
            sys.exit('Unknown precisions specified: {}.'.format(', '.join(sorted(unknown_precisions))))

    topologies = common.load_topologies_from_args(parser, args)

    output_dir = args.download_dir if args.output_dir is None else args.output_dir

    def convert(top, do_prefix_stdout=True):
        stdout_prefix = None
        if do_prefix_stdout:
            stdout_prefix = threading.current_thread().name

        if top.mo_args is None:
            prefixed_printf(stdout_prefix, '========= Skipping {} (no conversions defined)', top.name)
            prefixed_printf(stdout_prefix, '')
            return True

        top_precisions = requested_precisions & top.precisions
        if not top_precisions:
            prefixed_printf(stdout_prefix, '========= Skipping {} (all conversions skipped)', top.name)
            prefixed_printf(stdout_prefix, '')
            return True

        top_format = top.framework

        if top.pytorch_to_onnx_args:
            if not convert_to_onnx(top, output_dir, args, stdout_prefix):
                return False
            top_format = 'onnx'

        expanded_mo_args = [
            string.Template(arg).substitute(dl_dir=args.download_dir / top.subdirectory,
                                            mo_dir=mo_path.parent,
                                            conv_dir=output_dir / top.subdirectory)
            for arg in top.mo_args]

        for top_precision in top_precisions:
            mo_cmd = [str(args.python), '--', str(mo_path),
                '--framework={}'.format(top_format),
                '--data_type={}'.format(top_precision),
                '--output_dir={}'.format(output_dir / top.subdirectory / top_precision),
                '--model_name={}'.format(top.name),
                *expanded_mo_args]

            prefixed_printf(stdout_prefix, '========= {}Converting {} to IR ({})',
                '(DRY RUN) ' if args.dry_run else '', top.name, top_precision)

            prefixed_printf(stdout_prefix, 'Conversion command: {}', ' '.join(map(quote_arg, mo_cmd)))

            if not args.dry_run:
                prefixed_printf(stdout_prefix, '', flush=True)

                if not prefixed_subprocess(stdout_prefix, mo_cmd):
                    return False

            prefixed_printf(stdout_prefix, '')

        return True

    if args.jobs == 1:
        results = [convert(top, do_prefix_stdout=False) for top in topologies]
    else:
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as executor:
            results = list(executor.map(convert, topologies))

    failed_topologies = [top.name for top, successful in zip(topologies, results) if not successful]

    if failed_topologies:
        print('FAILED:')
        print(*sorted(failed_topologies), sep='\n')
        sys.exit(1)

if __name__ == '__main__':
    main()
