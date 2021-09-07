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

import importlib
import os
import re
import sys
import warnings
import platform
import subprocess
from setuptools import find_packages, setup
from setuptools.command.test import test as test_command
from setuptools.command.install import install as install_command
from distutils.version import LooseVersion
from pathlib import Path

here = Path(__file__).parent


class PyTest(test_command):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        test_command.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import shlex
        # import here, cause outside the eggs aren't loaded
        import pytest

        error_code = pytest.main(shlex.split(self.pytest_args))
        sys.exit(error_code)


def read(*path):
    input_file = os.path.join(here, *path)
    with open(str(input_file), encoding='utf-8') as file:
        return file.read()


def check_and_update_numpy(min_acceptable='1.15'):
    try:
        import numpy as np
        update_required = LooseVersion(np.__version__) < LooseVersion(min_acceptable)
    except ImportError:
        update_required = True
    if update_required:
        subprocess.call([sys.executable, '-m', 'pip', 'install', 'numpy>={}'.format(min_acceptable)])


def install_dependencies_with_pip(dependencies):
    for dep in dependencies:
        if dep.startswith('#'):
            continue
        subprocess.call([sys.executable, '-m', 'pip', 'install', str(dep)])


class CoreInstall(install_command):
    pass


def find_version(*path):
    version_file = read(*path)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)

    raise RuntimeError("Unable to find version string.")

is_arm = platform.processor() == 'aarch64'
long_description = read("README.md")
version = find_version("accuracy_checker", "__init__.py")


def prepare_requirements():
    requirements_core = read('requirements-core.in').split('\n')
    if 'install_core' in sys.argv:
        return requirements_core
    requirements = read("requirements.in").split('\n')
    return requirements_core + requirements


requirements = prepare_requirements()

try:
    importlib.import_module('cv2')
except ImportError as opencv_import_error:
    if platform.processor() != 'aarch64':
        warnings.warn(
            "Problem with cv2 import: \n{}\n opencv-python will be added to requirements".format(opencv_import_error)
        )
        requirements.append('opencv-python')
    else:
        warnings.warn("Problem with cv2 import: \n{}.\n Probably due to unsuitable numpy version, will be updated".format(opencv_import_error))
        check_and_update_numpy()

if is_arm:
    install_dependencies_with_pip(requirements)

setup(
    name="accuracy_checker",
    description="Deep Learning Accuracy validation framework",
    version=version,
    long_description=long_description,
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "accuracy_check=accuracy_checker.main:main",
            "convert_annotation=accuracy_checker.annotation_converters.convert:main",
    ]},
    zip_safe=False,
    python_requires='>=3.5',
    install_requires=requirements if not is_arm else '',
    tests_require=[read("requirements-test.in")],
    cmdclass={'test': PyTest, 'install_core': CoreInstall}
)
