#!/usr/bin/env python3
#
# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This file is based in part on setup.py and build.py from https://github.com/parlance/ctcdecode,
# commit 431408f22d93ef5ebc4422995111bbb081b971a9 on Apr 4, 2020, 20:54:49 UTC+1.
#
import glob
import numpy
import os.path
import setuptools


compile_args = ['-O3', '-std=c++11', '-fPIC']
yoklm_includes = ['ctcdecode_numpy/yoklm']
yoklm_sources = glob.glob('ctcdecode_numpy/yoklm/*.cpp')
third_party_libs = ['ThreadPool']
third_party_includes = [os.path.realpath(os.path.join("third_party", lib)) for lib in third_party_libs]
ctc_sources = glob.glob('ctcdecode_numpy/*.cpp')

extension = setuptools.Extension(
    name='ctcdecode_numpy._impl',
    sources=ctc_sources + yoklm_sources,
    include_dirs=third_party_includes + yoklm_includes + [numpy.get_include()],
    extra_compile_args=compile_args,
    language='c++',
    swig_opts=['-c++'],
)

setuptools.setup(
    name='ctcdecode-numpy',
    version='0.3.0',
    description="CTC Decoder for NumPy based on implementation from PaddlePaddle-Deepspeech and Parlance ctcdecode",
    packages=['ctcdecode_numpy'],
    ext_modules=[extension],
    py_modules=['ctcdecode_numpy.impl'],
    install_requires=['numpy'],
)
