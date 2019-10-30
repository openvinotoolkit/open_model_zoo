// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include<iostream> // TODO: remove

#include <deque>

class MemoryMonitor {
public:
    MemoryMonitor();
    bool isEnabled() const;
    void enable(std::size_t  historySize);
    void disable();
    void collectData();
    std::deque<std::pair<double, double>> getLastHistory() const;
    double getMeanMem() const;
    double getMeanSwap() const;
    double getMaxMem() const;
    double getMaxSwap() const;

    const double memTotal, swapTotal; // in GiB
private:
    bool enabled;
    unsigned samplesNumber;
    std::size_t historySize;
    double meanMem, meanSwap;
    double maxMem, maxSwap;
    std::deque<std::pair<double, double>> memSwapUsageHistory;
};
