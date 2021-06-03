// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

class ExponentialAverager {
public:
    ExponentialAverager(double smoothingFactor, double initValue);
    double getAveragedValue() const;
    void updateValue(double newValue);

private:
    double smoothingFactor;
    double value;
};
