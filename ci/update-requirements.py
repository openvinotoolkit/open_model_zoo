#!/usr/bin/env python3

"""
This script updates all of the requirements-*.txt files in this directory
with the most recent package versions.

It uses pip-compile (https://github.com/jazzband/pip-tools), so install that
before running it.
"""

import os
import subprocess
import sys

from pathlib import Path

# Package dependencies can vary depending on the Python version.
# We thus have to run pip-compile with the lowest Python version that
# the project supports.
EXPECTED_PYTHON_VERSION = (3, 5)

repo_root = Path(__file__).resolve().parent.parent

def pip_compile(target, *sources):
    print('updating {}...'.format(target), flush=True)

    # Use --no-header, since the OpenVINO install path may vary between machines,
    # so it should not be embedded in the output file. Also, this script makes
    # the information in pip-compile's headers redundant.

    subprocess.run(
        [sys.executable, '-mpiptools', 'compile',
            '--no-header', '--upgrade', '--quiet', '-o', target, '--', *map(str, sources)],
        check=True, cwd=str(repo_root))

def main():
    if sys.version_info[:2] != EXPECTED_PYTHON_VERSION:
        sys.exit("run this with Python {}".format('.'.join(map(str, EXPECTED_PYTHON_VERSION))))

    if 'INTEL_OPENVINO_DIR' not in os.environ:
        sys.exit("run OpenVINO toolkit's setupvars.sh before this")

    openvino_dir = Path(os.environ['INTEL_OPENVINO_DIR'])

    pip_compile('ci/requirements-ac.txt',
        'tools/accuracy_checker/requirements.in')
    pip_compile('ci/requirements-ac-test.txt',
        'tools/accuracy_checker/requirements.in', 'tools/accuracy_checker/requirements-test.in')
    pip_compile('ci/requirements-conversion.txt',
        'tools/downloader/requirements-pytorch.in', 'tools/downloader/requirements-caffe2.in',
        openvino_dir / 'deployment_tools/model_optimizer/requirements.txt')
    pip_compile('ci/requirements-demos.txt',
        'demos/python_demos/requirements.txt', openvino_dir / 'python/requirements.txt')
    pip_compile('ci/requirements-downloader.txt',
        'tools/downloader/requirements.in')

if __name__ == '__main__':
    main()
