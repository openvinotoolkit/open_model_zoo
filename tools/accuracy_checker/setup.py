"""
Copyright (c) 2019 Intel Corporation

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
import re
import sys
import warnings
from setuptools import find_packages, setup
from setuptools.command.test import test as test_command
from setuptools.command.install import install as install_command
from pathlib import Path


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
    version_file = Path(__file__).parent.joinpath(*path)
    with version_file.open() as file:
        return file.read()


class CoreInstall(install_command):
    pass


def find_version(*path):
    version_file = read(*path)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)

    raise RuntimeError("Unable to find version string.")


long_description = read("README.md")
version = find_version("accuracy_checker", "__init__.py")

requirements = [read('requirements-core.in') + read("requirements.in") if 'install_core' not in sys.argv else '']

try:
    importlib.import_module('cv2')
except ImportError as opencv_import_error:
    warnings.warn(
        "Problem with cv2 import: \n{}\n opencv-python will be added to requirements".format(opencv_import_error)
    )
    requirements.append('opencv-python')

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
    install_requires=requirements,
    tests_require=[read("requirements-test.in")],
    cmdclass={'test': PyTest, 'install_core': CoreInstall}
)
