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
    void enableHistory(std::size_t historySize); // makes getLastHistory() store at least historySize
    bool isHistoryEnabled() const;
    void disableHistory();
    void enableLast();
    bool isLastEnabled() const;
    void disableLast();
    void collectData();
    std::deque<std::vector<double>> getLastHistory() const;
    std::vector<double> getMeanCpuLoad() const;

    const std::size_t nCores;
private:
    bool lastEnabled, historyEnabled;
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
