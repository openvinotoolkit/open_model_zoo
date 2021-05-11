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

import distutils.command.build
import distutils.command.sdist
import itertools
import os
import shutil
from pathlib import Path

from setuptools import setup

SETUP_DIR = Path(__file__).resolve().parent

# When pip builds a project, it copies the project directory into a temporary
# directory. Therefore, if we try to access OMZ files by just looking in the
# parent directories, we won't find them. To permit building with pip, allow
# the user to manually specify the OMZ directory location.
# This is optional, because if the project is built directly with setup.py,
# we can access the parent directories normally.
# This hack might become unnecessary if https://github.com/pypa/pip/issues/7555
# is resolved.
OMZ_ROOT = Path(os.environ.get('OMZ_ROOT', SETUP_DIR.parents[1]))

def read_text(path):
    return (SETUP_DIR / path).read_text()

# We can't build sdists, because we depend on files outside the
# project directory. Disable the sdist command to prevent creation
# of broken sdists.
class DisabledSdist(distutils.command.sdist.sdist):
    def run(self):
        raise RuntimeError("building sdists is not supported")

class CustomBuild(distutils.command.build.build):
    def run(self):
        super().run()

        package_build_dir = Path(self.build_lib, 'open_model_zoo/model_tools')
        (package_build_dir / 'data').mkdir(exist_ok=True)

        shutil.copy(
            str(OMZ_ROOT / 'data/dataset_definitions.yml'),
            str(package_build_dir / 'data'),
        )

        for model_config_path in itertools.chain(
                OMZ_ROOT.glob('models/**/model.yml'),
                OMZ_ROOT.glob('models/**/composite-model.yml'),
        ):
            model_dir = model_config_path.parent

            for path in model_dir.glob('**/*'):
                if path.is_dir() and not path.is_symlink():
                    continue

                if path.suffix == '.md' or (model_dir / 'assets') in path.parents:
                    continue

                path_rel = path.relative_to(OMZ_ROOT)

                if path_rel.suffix in ('.png', '.jpg'):
                    # Fail-safe: if we're here, it means that either the repository
                    # layout was changed and this script wasn't updated, or that a
                    # model was not laid out correctly. Either way, we need to fail
                    # so that the problem can be corrected.
                    raise RuntimeError(f"unexpected file in 'models': {path_rel}")

                (package_build_dir / path_rel.parent).mkdir(exist_ok=True, parents=True)
                shutil.copy(str(path), str(package_build_dir / path_rel))

setup(
    install_requires=read_text('requirements.in'),
    extras_require={
        'caffe2': read_text('requirements-caffe2.in'),
        'pytorch': read_text('requirements-pytorch.in'),
        'tensorflow2': read_text('requirements-tensorflow.in'),
    },
    cmdclass={
        'build': CustomBuild,
        'sdist': DisabledSdist,
    },
)
