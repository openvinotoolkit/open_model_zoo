// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdio>
#include <string>

#include "exponential_averager.hpp"

ExponentialAverager::ExponentialAverager(double smoothingFactor, double initValue):
                     smoothingFactor(smoothingFactor), value(initValue) {
}

double ExponentialAverager::getAveragedValue() const {
    return value;
}

void ExponentialAverager::updateValue(double newValue) {
    value = smoothingFactor * newValue + (1. - smoothingFactor) * value;
}

