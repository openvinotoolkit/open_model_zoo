// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "perf_timer.hpp"

PerfTimer::PerfTimer(size_t maxCount_):
    maxCount(maxCount_) {
    values.reserve(maxCount);
}

float PerfTimer::getValue() const {
    return avgValue;
}

bool PerfTimer::enabled() const {
    return maxCount > 0;
}
