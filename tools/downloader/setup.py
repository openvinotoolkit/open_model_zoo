#!/usr/bin/env python3

# Copyright (c) 2021 Intel Corporation
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
This setup script is primarily intended to be used to package the model
tools as part of the openvino-dev distribution. This is why it (along
with setup.cfg) specifies only the bare minimum of distribution metadata.

If you're an end user, you most likely don't need to run this. Just execute
the scripts in this directory (downloader.py, etc.) directly.
"""

from pathlib import Path

from setuptools import setup

SETUP_DIR = Path(__file__).parent

def read_text(path):
    return (SETUP_DIR / path).read_text()

setup(
    install_requires=read_text('requirements.in'),
    extras_require={
        'caffe2': read_text('requirements-caffe2.in'),
        'pytorch': read_text('requirements-pytorch.in'),
        'tensorflow2': read_text('requirements-tensorflow.in'),
    },
)
