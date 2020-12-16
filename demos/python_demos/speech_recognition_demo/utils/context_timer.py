#
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import timeit


class Timer:
    def __enter__(self):
        self.elapsed = None
        self.start = timeit.default_timer()
        return self

    def __exit__(self, type, value, traceback):
        self.end = timeit.default_timer()
        self.elapsed = self.end - self.start
