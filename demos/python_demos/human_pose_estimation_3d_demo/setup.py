#!/usr/bin/env python
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

import os
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sys


PACKAGE_NAME = 'pose_extractor'


options = {'--debug': 'OFF'}
if '--debug' in sys.argv:
    options['--debug'] = 'ON'


class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir=PACKAGE_NAME, **kwargs):
        Extension.__init__(self, name, sources=[], **kwargs)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class CMakeBuild(build_ext):
    def build_extensions(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError('Cannot find CMake executable')

        ext = self.extensions[0]
        build_dir = os.path.abspath(os.path.join(PACKAGE_NAME, 'build'))
        if not os.path.exists(build_dir):
            os.mkdir(build_dir)
        tmp_dir = os.path.join(build_dir, 'tmp')
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)

        cfg = 'Debug' if options['--debug'] == 'ON' else 'Release'

        cmake_args = [
            '-DCMAKE_BUILD_TYPE={}'.format(cfg),
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), build_dir),
            '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), tmp_dir),
            '-DPYTHON_EXECUTABLE={}'.format(sys.executable)
        ]

        if platform.system() == 'Windows':
            platform_type = ('x64' if platform.architecture()[0] == '64bit' else 'Win32')
            cmake_args += [
                '-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE',
                '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), build_dir),
            ]

            if self.compiler.compiler_type == 'msvc':
                cmake_args += [
                    '-DCMAKE_GENERATOR_PLATFORM={}'.format(platform_type),
                ]
            else:
                cmake_args += [
                    '-G', 'MinGW Makefiles',
                ]

        subprocess.check_call(['cmake', ext.cmake_lists_dir] + cmake_args, cwd=tmp_dir)
        subprocess.check_call(['cmake', '--build', '.', '--config', cfg], cwd=tmp_dir)


setup(name=PACKAGE_NAME,
      packages=[PACKAGE_NAME],
      version='1.0',
      description='Auxiliary C++ module for fast 2d pose extraction from network output',
      ext_modules=[CMakeExtension(PACKAGE_NAME)],
      cmdclass={'build_ext': CMakeBuild})
