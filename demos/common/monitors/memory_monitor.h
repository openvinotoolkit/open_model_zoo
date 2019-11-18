// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include<iostream> // TODO: remove

#include <deque>

class MemoryMonitor {
public:
    MemoryMonitor();
    void setHistorySize(std::size_t size);
    std::size_t getHistorySize() const;
    void collectData();
    std::deque<std::pair<double, double>> getLastHistory() const;
    double getMeanMem() const; // in GiB
    double getMeanSwap() const;
    double getMaxMem() const;
    double getMaxSwap() const;
    double getMemTotal() const;
    double getMaxMemTotal() const; // a system may have hotpluggable memory
private:
    bool enabled;
    unsigned samplesNumber;
    std::size_t historySize;
    double memSum, swapSum;
    double maxMem, maxSwap;
    double memTotal, swapTotal;
    double maxMemTotal, maxSwapTotal;
    std::deque<std::pair<double, double>> memSwapUsageHistory;
};
