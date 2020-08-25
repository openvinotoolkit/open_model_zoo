#!/usr/bin/env python3
#
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This file is based in part on setup.py from https://github.com/parlance/ctcdecode,
# commit 431408f22d93ef5ebc4422995111bbb081b971a9 on Apr 4, 2020, 20:54:49 UTC+1.
#
import multiprocessing.pool
import os

from setuptools import setup, find_packages, distutils
from distutils.command.build import build as Build

this_file = os.path.dirname(__file__)


# monkey-patch for parallel compilation
# See: https://stackoverflow.com/a/13176803
def parallelCCompile(self,
                     sources,
                     output_dir=None,
                     macros=None,
                     include_dirs=None,
                     debug=0,
                     extra_preargs=None,
                     extra_postargs=None,
                     depends=None):
    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    # parallel code
    def _single_compile(obj):
        try:
            src, ext = build[obj]
        except KeyError:
            return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    # convert to list, imap is evaluated on-demand
    thread_pool = multiprocessing.pool.ThreadPool(12)
    list(thread_pool.imap(_single_compile, objects))
    return objects


# hack compile to support parallel compiling
distutils.ccompiler.CCompiler.compile = parallelCCompile


# Fix the problem with SWIG generated .py interface file appearing too late in the build command sequence.
# (But this fix doesn't work.  So go with `python setup.py build_ext install`.)
class BuildExtFirst(Build):
    sub_commands = (
        [('build_ext', dict(Build.sub_commands)['build_ext'])] +
        [cmd for cmd in Build.sub_commands if cmd[0] != 'build_ext']
    )


import build as build_extension

setup(
    name='ctcdecode-numpy',
    version='0.1',
    description="CTC Decoder for NumPy based on implementation from PaddlePaddle-Deepspeech and Parlance ctcdecode",
    packages=['ctcdecode_numpy'],
    ext_modules = [build_extension.extension],
    py_modules = ['ctcdecode_numpy.impl', 'ctcdecode_numpy._impl'],
    cmdclass = dict(build = BuildExtFirst),
)
