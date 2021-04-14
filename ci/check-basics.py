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

RE_SHEBANG_LINE = re.compile(r'\#! \s* (\S+) (?: \s+ (\S+))? \s*', re.VERBOSE)

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

    print('running miscellaneous checks...', flush=True)

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

        # Overly long paths cause problems in internal infrastructure due to Windows's
        # 259 character path length limit. Cap the repo path lengths at a known safe value.
        PATH_LENGTH_LIMIT = 135
        if len(path) > PATH_LENGTH_LIMIT:
            complain(f"{path}: path length ({len(path)}) over limit ({PATH_LENGTH_LIMIT})")

        mode = raw_diff.split()[1]

        absolute_path = OMZ_ROOT / path

        if path.startswith('tools/accuracy_checker/configs/') and path.endswith('.yml'):
            if mode == '120000':
                try:
                    if absolute_path.is_symlink():
                        real_path = absolute_path.resolve(strict=True)
                    else:
                        with open(absolute_path, 'r', newline='') as file:
                            link_target = file.read()
                        real_path = (absolute_path.parent / link_target).resolve(strict=True)
                except FileNotFoundError:
                    complain(f"{path}: should be a symbolic link to existing accuracy-check.yml from models directory")
                else:
                    model_name = absolute_path.stem
                    if real_path.name != 'accuracy-check.yml' or real_path.parent.name != model_name:
                        complain(f"{path}: should be a symbolic link to accuracy-check.yml from {model_name} model "
                                 "directory")
            else:
                complain(f"{path}: isn't a symbolic link but it should be a symbolic link to accuracy-check.yml "
                         "from models directory")

        if path.startswith('models/') and '/description/' in path:
            complain(f"{path}: the model documentation convention has changed;"
                " put the text in README.md and the images under /assets/")

        if mode not in {'100644', '100755'}: # not a regular or executable file
            continue

        if num_added_lines == '-': # binary file
            continue

        if path.startswith('demos/thirdparty/'):
            continue

        with open(absolute_path, encoding='UTF-8') as f:
            lines = list(f)

        if lines and not lines[-1].endswith('\n'):
            complain(f"{path}:{len(lines)}: last line doesn't end with a newline character")

        has_shebang = lines and lines[0].startswith('#!')
        is_executable = mode == '100755'

        if is_executable and not has_shebang:
            complain(f"{path}: is executable, but doesn't have a shebang line")

        if has_shebang:
            if not is_executable:
                complain(f"{path}: has a shebang line, but isn't executable")

            shebang_program, shebang_args = \
                RE_SHEBANG_LINE.fullmatch(lines[0].rstrip('\n')).groups()

            # Python 2 is EOL and OpenVINO doesn't support it anymore. If someone
            # uses `#!/usr/bin/python` or something similar, it's likely a mistake.
            if shebang_program.endswith('/python') or (
                    shebang_program.endswith('/env') and shebang_args == 'python'):
                complain(f"{path}:1: use 'python3', not 'python'")

    print('running yamllint...', flush=True)
    if subprocess.run([sys.executable, '-m', 'yamllint', '-s', '.'], cwd=OMZ_ROOT).returncode != 0:
        all_passed = False

    print('running flake8...', flush=True)
    if subprocess.run([sys.executable, '-m', 'flake8', '--config=.flake8'], cwd=OMZ_ROOT).returncode != 0:
        all_passed = False

    print('running documentation checks...', flush=True)
    if subprocess.run([sys.executable, '--', str(OMZ_ROOT / 'ci/check-documentation.py')]).returncode != 0:
        all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
