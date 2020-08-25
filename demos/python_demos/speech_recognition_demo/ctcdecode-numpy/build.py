#
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This file is based in part on build.py from https://github.com/parlance/ctcdecode,
# commit 431408f22d93ef5ebc4422995111bbb081b971a9 on Apr 4, 2020, 20:54:49 UTC+1.
#
import glob
import os
import tarfile
import warnings

import wget
import setuptools


def download_extract(url, dl_path):
    if not os.path.isfile(dl_path):
        # Already downloaded
        wget.download(url, out=dl_path)
    if dl_path.endswith(".tar.gz") and os.path.isdir(dl_path[:-len(".tar.gz")]):
        # Already extracted
        return
    tar = tarfile.open(dl_path)
    tar.extractall('third_party/')
    tar.close()


# Download/Extract openfst, boost
download_extract('https://sites.google.com/site/openfst/home/openfst-down/openfst-1.6.7.tar.gz',
                 'third_party/openfst-1.6.7.tar.gz')

for file in ['third_party/ThreadPool/ThreadPool.h']:
    if not os.path.exists(file):
        warnings.warn('File `{}` does not appear to be present. Did you forget `git submodule update`?'.format(file))


# Does gcc compile with this header and library?
def compile_test(header, library):
    dummy_path = os.path.join(os.path.dirname(__file__), "dummy")
    command = "bash -c \"g++ -include " + header + " -l" + library + " -x c++ - <<<'int main() {}' -o " + dummy_path \
              + " >/dev/null 2>/dev/null && rm " + dummy_path + " 2>/dev/null\""
    return os.system(command) == 0


compile_args = ['-O3', '-std=c++11', '-fPIC']
compile_args += ['-Wno-sign-compare', '-Wno-unused-local-typedefs']

yoklm_includes = ['ctcdecode_numpy/yoklm']
yoklm_sources = glob.glob('ctcdecode_numpy/yoklm/*.cpp')

third_party_libs = ['openfst-1.6.7/src/include', 'ThreadPool']
lib_sources = glob.glob('third_party/openfst-1.6.7/src/lib/*.cc')
lib_sources = [fn for fn in lib_sources if not (fn.endswith('main.cc') or fn.endswith('test.cc'))]

third_party_includes = [os.path.realpath(os.path.join("third_party", lib)) for lib in third_party_libs]
ctc_sources = glob.glob('ctcdecode_numpy/*.cpp') + ['ctcdecode_numpy/decoders.i'] + yoklm_sources
ctc_sources = list(filter(lambda fname: not fname.endswith('/decoders_wrap.cpp'), ctc_sources))


extension = setuptools.Extension(
   name='ctcdecode_numpy._impl',
   sources=ctc_sources + lib_sources,
   include_dirs=third_party_includes + yoklm_includes,
   extra_compile_args=compile_args,
   language='c++',
   swig_opts=['-c++'],
)
