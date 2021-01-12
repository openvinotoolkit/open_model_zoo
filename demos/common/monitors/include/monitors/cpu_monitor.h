// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <deque>
#include <memory>
#include <vector>

class CpuMonitor {
public:
    CpuMonitor();
    ~CpuMonitor();
    void setHistorySize(std::size_t size);
    std::size_t getHistorySize() const;
    void collectData();
    std::deque<std::vector<double>> getLastHistory() const;
    std::vector<double> getMeanCpuLoad() const;

private:
    unsigned samplesNumber;
    unsigned historySize;
    std::vector<double> cpuLoadSum;
    std::deque<std::vector<double>> cpuLoadHistory;
    class PerformanceCounter;
    std::unique_ptr<PerformanceCounter> performanceCounter;
};
