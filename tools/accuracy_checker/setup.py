"""
Copyright (c) 2018-2023 Intel Corporation

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

import importlib
import os
import re
import sys
import warnings
import platform
import subprocess # nosec - disable B404:import-subprocess check
from distutils.version import LooseVersion
from pathlib import Path
from setuptools import find_packages, setup # pylint:disable=W9902
from setuptools.command.test import test as test_command # pylint:disable=W9902
from setuptools.command.install import install as install_command # pylint:disable=W9902

here = Path(__file__).parent


class PyTest(test_command):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        test_command.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import shlex # pylint:disable=C0415
        # import here, cause outside the eggs aren't loaded
        import pytest # pylint:disable=C0415

        error_code = pytest.main(shlex.split(self.pytest_args))
        sys.exit(error_code)


def read(*path):
    input_file = os.path.join(here, *path)
    with open(str(input_file), encoding='utf-8') as file:
        return file.read()


def check_and_update_numpy(min_acceptable='1.15'):
    try:
        import numpy as np # pylint:disable=C0415
        update_required = LooseVersion(np.__version__) < LooseVersion(min_acceptable)
    except ImportError:
        update_required = True
    if update_required:
        subprocess.call([sys.executable, '-m', 'pip', 'install', 'numpy>={}'.format(min_acceptable)])


class CoreInstall(install_command):
    pass


def find_version(*path):
    version_file = read(*path)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)

    raise RuntimeError("Unable to find version string.")

long_description = read("README.md")
version = find_version("openvino/tools/accuracy_checker", "__init__.py")


def prepare_requirements():
    requirements_core = read('requirements-core.in').split('\n')
    if 'install_core' in sys.argv:
        warnings.warn(
            '"install_core" command is deprecated and will be removed in 2023.1 release, please use "install" instead',
            DeprecationWarning
        )
    requirements = read("requirements-extra.in").split('\n')
    return requirements_core, requirements


_requirements, _extras = prepare_requirements()

try:
    importlib.import_module('cv2')
except ImportError as opencv_import_error:
    if platform.processor() != 'aarch64':
        warnings.warn(
            "Problem with cv2 import: \n{}\n opencv-python will be added to requirements".format(opencv_import_error)
        )
        _requirements.append('opencv-python')
    else:
        warnings.warn(
            "Problem with cv2 import: \n{}".format(opencv_import_error)
            + "\n Probably due to unsuitable numpy version, will be updated")
        check_and_update_numpy()


setup(
    name="accuracy_checker",
    description="Deep Learning Accuracy validation framework",
    version=version,
    long_description=long_description,
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "accuracy_check=openvino.tools.accuracy_checker.main:main",
            "convert_annotation=openvino.tools.accuracy_checker.annotation_converters.convert:main"]},
    zip_safe=False,
    python_requires='>=3.5',
    install_requires=_requirements,
    tests_require=[read("requirements-test.in")],
    cmdclass={'test': PyTest, 'install_core': CoreInstall},
    extras_require={'extra': _extras + ['pycocotools>=2.0.2', 'crf_beam;platform_system=="Linux"', 'torch>=0.4.0', 'torchvision>=0.2.1', 'lpips', 'soundfile', "torchmetrics", "diffusers",
                              'kenlm @ git+https://github.com/kpu/kenlm.git@f01e12d83c7fd03ebe6656e0ad6d73a3e022bd50#egg=kenlm']}
)
