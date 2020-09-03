#!/usr/bin/env python3

# Copyright (c) 2020 Intel Corporation
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

"""
This script performs various checks on the source code that can be run quickly.
The idea is that these are the kind of checks that can be run on every pull
request without substantially impacting build time.
"""

import re
import subprocess
import sys

from pathlib import Path


OMZ_ROOT = Path(__file__).resolve().parents[1]

def main():
    all_passed = True

    def complain(message):
        nonlocal all_passed
        all_passed = False
        print(message, file=sys.stderr)

    # get the Git magic empty tree hash
    empty_tree_hash = subprocess.run(
        ['git', 'hash-object', '-t', 'tree', '--stdin'],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        check=True,
        universal_newlines=True,
        cwd=OMZ_ROOT,
    ).stdout.strip()

    if subprocess.run(['git', '--no-pager', 'diff', '--check', empty_tree_hash, '--'],
            cwd=OMZ_ROOT).returncode != 0:
        all_passed = False

    raw_diff_lines = subprocess.run(
        ['git', 'diff', '--raw', '-z', empty_tree_hash, '--'],
        check=True,
        stdout=subprocess.PIPE,
        cwd=OMZ_ROOT,
    ).stdout.decode()[:-1].split('\0')

    numstat_lines = subprocess.run(
        ['git', 'diff', '--numstat', '-z', empty_tree_hash, '--'],
        check=True,
        stdout=subprocess.PIPE,
        cwd=OMZ_ROOT,
    ).stdout.decode()[:-1].split('\0')

    for raw_diff, path, numstat_line in zip(*[iter(raw_diff_lines)] * 2, numstat_lines):
        num_added_lines, num_removed_lines, numstat_path = numstat_line.split('\t', 2)
        assert path == numstat_path, "git diff --raw and --numstat list different files"

        mode = raw_diff.split()[1]

        if mode not in {'100644', '100755'}: # not a regular or executable file
            continue

        if num_added_lines == '-': # binary file
            continue

        if path.startswith('demos/thirdparty/'):
            continue

        with open(OMZ_ROOT / path, encoding='UTF-8') as f:
            lines = list(f)

        if lines and not lines[-1].endswith('\n'):
            complain(f"{path}:{len(lines)}: last line doesn't end with a newline character")

        has_shebang = lines and lines[0].startswith('#!')
        is_executable = mode == '100755'

        if is_executable and not has_shebang:
            complain(f"{path}: is executable, but doesn't have a shebang line")

        if has_shebang and not is_executable:
            complain(f"{path}: has a shebang line, but isn't executable")

    if subprocess.run([sys.executable, '-m', 'yamllint', '-s', '.'], cwd=OMZ_ROOT).returncode != 0:
        all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
