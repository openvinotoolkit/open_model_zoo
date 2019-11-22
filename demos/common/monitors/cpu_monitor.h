// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream> // TODO remove
#include <chrono>
#include <deque>
#include <vector>

class CpuMonitor {
public:
    CpuMonitor();
    void setHistorySize(std::size_t size);
    std::size_t getHistorySize() const;
    void collectData();
    std::deque<std::vector<double>> getLastHistory() const;
    std::vector<double> getMeanCpuLoad() const;

    const std::size_t nCores;
private:
    bool lastEnabled;
    unsigned samplesNumber;
    unsigned historySize;
    std::vector<double> cpuLoadSum;
    std::deque<std::vector<double>> cpuLoadHistory;
    std::vector<unsigned long> prevIdleCpuStat;
    std::chrono::steady_clock::time_point prevTimePoint;

#ifdef _WIN32
public:
    ~CpuMonitor();
private:
    void openQuery();
    void closeQuery();

    struct PerformanceCounter;
    std::unique_ptr<PerformanceCounter> performanceCounter;
#endif
};
