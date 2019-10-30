// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream> // TODO remove
#include <deque>
#include <vector>
#ifdef WIN32
#include <windows.h>
#include <Sysinfoapi.h>
#include <pdh.h>
#endif

class CpuMonitor {
public:
    CpuMonitor();
    void enableHistory(std::size_t historySize); // makes getLastHistory() store at least historySize
    bool isHistoryEnabled();
    void disableHistory();
    void enableLast();
    bool isLastEnabled();
    void disableLast();
    void collectData();
    std::deque<std::vector<double>> getLastHistory();
    std::vector<double> getMeanCpuLoad(); // can be empty if monitor didn't collect any data

    const std::size_t nCores;
private:
    bool lastEnabled, historyEnabled;
    unsigned samplesNumber;
    unsigned historySize;
    std::vector<double> meanCpuLoad;
    std::deque<std::vector<double>> cpuLoadHistory;
    std::vector<std::pair<unsigned long, unsigned long>> prevIdleNonIdleCpuStat;

#ifdef WIN32
    void openQuery();
    void closeQuery();

    PDH_HQUERY query;
    std::vector<PDH_HCOUNTER> coreTimeCounters;
#endif
};
