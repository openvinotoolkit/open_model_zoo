#!/usr/bin/env python3
"""
 Copyright (c) 2021-2024 Intel Corporation

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

"""
Use this script to create a wheel for Open Model Zoo
model API. The installation of wheel is described in
`<omz_dir>/demos/README.md`
"""

from pathlib import Path
from setuptools import setup, find_packages


SETUP_DIR = Path(__file__).resolve().parent

with open(SETUP_DIR / 'requirements.txt') as f:
    required = f.read().splitlines()

with open(SETUP_DIR / 'requirements_ovms.txt') as f:
    ovms_required = f.read().splitlines()

packages = find_packages(str(SETUP_DIR))
packages.remove('visualizers')

setup(
    name='openmodelzoo-modelapi',
    version='0.0.0',
    author='Intel® Corporation',
    license='OSI Approved :: Apache Software License',
    url='https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/common/python/model_zoo/model_api',
    description='Model API: model wrappers and pipelines from Open Model Zoo',
    python_requires = ">=3.7",
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
    ],
    packages=packages,
    install_requires=required,
    extras_require={'ovms': ovms_required}
)
