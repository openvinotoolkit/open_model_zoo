// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream> // TODO remove
#include <deque>
#include <vector>
#ifdef _WIN32
#include <pdh.h>
#endif

class CpuMonitor {
public:
    CpuMonitor();
    void setHistorySize(std::size_t historySize);
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
    std::vector<std::pair<unsigned long, unsigned long>> prevIdleNonIdleCpuStat;

#ifdef _WIN32
    void openQuery();
    void closeQuery();

    PDH_HQUERY query;
    std::vector<PDH_HCOUNTER> coreTimeCounters;
#endif
};
