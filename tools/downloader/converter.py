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
import os
import platform
import re
import shlex
import string
import subprocess
import sys

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, metavar='CONFIG.YML',
        default=common.get_default_config_path(), help='topology configuration file')
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
    parser.add_argument('-p', '--python', type=Path, metavar='PYTHON', default=sys.executable,
        help='Python executable to run Model Optimizer with')
    parser.add_argument('--mo', type=Path, metavar='MO.PY',
        help='Model Optimizer entry point script')
    parser.add_argument('--dry-run', action='store_true',
        help='Print the conversion commands without running them')
    args = parser.parse_args()

    mo_path = args.mo
    if mo_path is None:
        try:
            mo_path = Path(os.environ['INTEL_OPENVINO_DIR']) / 'deployment_tools/model_optimizer/mo.py'
        except KeyError:
            sys.exit('Unable to locate Model Optimizer. '
                + 'Use --mo or run setupvars.sh/setupvars.bat from the OpenVINO toolkit.')

    topologies = common.load_topologies_from_args(parser, args)

    output_dir = args.download_dir if args.output_dir is None else args.output_dir

    failed_topologies = set()

    for top in topologies:
        if top.mo_args is None:
            print('========= Skipping {} (no conversion defined)'.format(top.name))
            print()
            continue

        expanded_mo_args = [
            string.Template(arg).substitute(dl_dir=args.download_dir / top.subdirectory, mo_dir=mo_path.parent)
            for arg in top.mo_args]

        assert len(top.precisions) == 1 # only one precision per model is supported at the moment

        mo_cmd = [str(args.python), '--', str(mo_path),
            '--output_dir={}'.format(output_dir / top.subdirectory / next(iter(top.precisions))),
            '--model_name={}'.format(top.name),
            *expanded_mo_args]

        print('========= {}Converting {}'.format('(DRY RUN) ' if args.dry_run else '', top.name))

        print('Conversion command:', ' '.join(map(quote_arg, mo_cmd)))

        if not args.dry_run:
            print(flush=True)

            if subprocess.run(mo_cmd).returncode != 0:
                failed_topologies.add(top.name)

        print()

    if failed_topologies:
        print('FAILED:')
        print(*sorted(failed_topologies), sep='\n')
        sys.exit(1)

if __name__ == '__main__':
    main()
