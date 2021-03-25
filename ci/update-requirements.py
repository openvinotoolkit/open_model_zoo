#!/usr/bin/env python3

"""
This script updates all of the requirements-*.txt files in this directory
with the most recent package versions.

It uses pip-compile (https://github.com/jazzband/pip-tools), so install that
before running it.
"""

import argparse
import os
import subprocess
import sys

from pathlib import Path

# Package dependencies can vary depending on the Python version.
# We thus have to run pip-compile with the lowest Python version that
# the project supports.
EXPECTED_PYTHON_VERSION = (3, 6)

repo_root = Path(__file__).resolve().parent.parent
script_name = Path(__file__).name

def fixup_req_file(req_path, path_placeholders):
    contents = req_path.read_text()

    for path, placeholder in path_placeholders:
        contents = contents.replace(f'-r {path}/', f'-r ${{{placeholder}}}/')
        contents = contents.replace(f'({path}/', f'(${{{placeholder}}}/')

    contents = "# use {} to update this file\n\n".format(script_name) + contents
    req_path.write_text(contents)

def pip_compile(target, *sources, upgrade=False):
    print('updating {}...'.format(target), flush=True)

    # Use --no-header, since the OpenVINO install path may vary between machines,
    # so it should not be embedded in the output file. Also, this script makes
    # the information in pip-compile's headers redundant.

    subprocess.run(
        [sys.executable, '-mpiptools', 'compile',
            *(['--upgrade'] if upgrade else []),
            '--no-header', '--quiet', '-o', target, '--', *map(str, sources)],
        check=True, cwd=str(repo_root))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upgrade', action='store_true', help='Bump package versions')
    args = parser.parse_args()

    if sys.version_info[:2] != EXPECTED_PYTHON_VERSION:
        sys.exit("run this with Python {}".format('.'.join(map(str, EXPECTED_PYTHON_VERSION))))

    if 'INTEL_OPENVINO_DIR' not in os.environ:
        sys.exit("run OpenVINO toolkit's setupvars.sh before this")

    openvino_dir = Path(os.environ['INTEL_OPENVINO_DIR'])

    def pc(target, *sources):
        pip_compile(target, *sources, upgrade=args.upgrade)
        fixup_req_file(repo_root / target, [(openvino_dir, 'INTEL_OPENVINO_DIR')])

    pc('ci/requirements-ac.txt',
        'tools/accuracy_checker/requirements-core.in', 'tools/accuracy_checker/requirements.in')
    pc('ci/requirements-ac-test.txt',
        'tools/accuracy_checker/requirements.in', 'tools/accuracy_checker/requirements-test.in',
        'tools/accuracy_checker/requirements-core.in')
    pc('ci/requirements-check-basics.txt',
       'ci/requirements-check-basics.in', 'ci/requirements-documentation.in')
    pc('ci/requirements-conversion.txt',
        *(f'tools/downloader/requirements-{suffix}.in' for suffix in ['caffe2', 'pytorch', 'tensorflow']),
        *(openvino_dir / f'deployment_tools/model_optimizer/requirements_{suffix}.txt'
            for suffix in ['caffe', 'mxnet', 'onnx', 'tf2']))
    pc('ci/requirements-demos.txt',
        'demos/requirements.txt', openvino_dir / 'python/requirements.txt')
    pc('ci/requirements-downloader.txt',
        'tools/downloader/requirements.in')
    pc('ci/requirements-quantization.txt',
        'tools/accuracy_checker/requirements-core.in', 'tools/accuracy_checker/requirements.in',
        openvino_dir / 'deployment_tools/tools/post_training_optimization_toolkit/setup.py',
        openvino_dir / 'deployment_tools/model_optimizer/requirements_kaldi.txt')

if __name__ == '__main__':
    main()
