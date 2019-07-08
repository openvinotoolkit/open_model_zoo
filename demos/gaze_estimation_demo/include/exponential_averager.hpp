// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>
#include <string>

class ExponentialAverager {
public:
    ExponentialAverager(double smoothingFactor, double initValue);
    double getAveragedValue() const;
    void updateValue(double newValue);

private:
    double smoothingFactor;
    double value;
};
