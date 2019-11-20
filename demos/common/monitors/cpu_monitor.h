// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream> // TODO remove
#include <deque>
#include <vector>
#ifdef _WIN32
#include <query_wrapper.h>
#endif
#include <chrono>

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
    void openQuery();
    void closeQuery();

    std::unique_ptr<QueryWrapper> query;
    std::vector<PDH_HCOUNTER> coreTimeCounters;
#endif
};
