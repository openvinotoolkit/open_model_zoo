// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_monitor.h"
#ifdef _WIN32
#include <windows.h>
#include <Sysinfoapi.h>
#include <tchar.h>

namespace {
std::size_t getNCores() {
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
}
}

CpuMonitor::CpuMonitor() :
    nCores{getNCores()},
    lastEnabled{false},
    samplesNumber{0},
    historySize{0},
    cpuLoadSum(nCores, 0) {}

void CpuMonitor::openQuery() {
    PDH_STATUS status;

    status = PdhOpenQuery(NULL, NULL, &query);
    if (ERROR_SUCCESS != status) {
        throw std::runtime_error("PdhOpenQuery() failed");
    }

    for (std::size_t i = 0; i < nCores; ++i)
    {
        TCHAR szCounterName[MAX_PATH];
        _stprintf_s(szCounterName, sizeof(szCounterName), TEXT("\\Processor(%u)\\%% Processor Time"), i);

        PDH_HCOUNTER counter;

        status = PdhAddCounter(query, szCounterName, 0, &counter);
        if (ERROR_SUCCESS != status)
        {
            throw std::runtime_error("PdhAddCounter() failed");
        }

        status = PdhSetCounterScaleFactor(counter, -2); // scale counter to [0, 1]
        if (ERROR_SUCCESS != status)
        {
            throw std::runtime_error("PdhSetCounterScaleFactor() failed");
        }

        coreTimeCounters.push_back(counter);
    }
    status = PdhCollectQueryData(query);
    if (ERROR_SUCCESS != status)
    {
        throw std::runtime_error("PdhCollectQueryData() failed");
    }
}

void CpuMonitor::closeQuery() {
    PDH_STATUS status = PdhCloseQuery(query);
    if (ERROR_SUCCESS != status) {
        throw std::runtime_error("PdhCloseQuery() failed");
    }
}

void CpuMonitor::enableHistory(std::size_t historySize) {
    if (historySize < 2) {
        disableHistory();
    }
    else {
        this->historySize = historySize;
        if (!lastEnabled) {
            openQuery();
        }
    }
}

bool CpuMonitor::isHistoryEnabled() const {
    return historySize > 1;
}

void CpuMonitor::disableHistory() {
    historySize = 1;
    if (!lastEnabled) {
        closeQuery();
    }
}

void CpuMonitor::enableLast() {
    lastEnabled = true;
    if (!isHistoryEnabled()) {
        historySize = 1;
        openQuery();
    }
}

bool CpuMonitor::isLastEnabled() const {
    return lastEnabled;
}

void CpuMonitor::disableLast() {
    lastEnabled = false;
    if (!isHistoryEnabled()) {
        closeQuery();
    }
}

void CpuMonitor::collectData() {
    // store CPU Utulization in range [0, 1]
    PDH_STATUS status;
    PDH_FMT_COUNTERVALUE DisplayValue;

    status = PdhCollectQueryData(query);
    if (ERROR_SUCCESS != status) {
        throw std::runtime_error("PdhCollectQueryData() failed");
    }

    std::vector<std::pair<double, double>> idleNonIdleCpuStat;
    for (std::size_t i = 0; i < coreTimeCounters.size(); i++) {
        status = PdhGetFormattedCounterValue(coreTimeCounters[i], PDH_FMT_DOUBLE, NULL, &DisplayValue);
        if (ERROR_SUCCESS != status) {
            throw std::runtime_error("PdhGetFormattedCounterValue() failed");
        }

        idleNonIdleCpuStat.emplace_back(DisplayValue.doubleValue, 1 - DisplayValue.doubleValue);
    }

    std::vector<double> cpuLoad(idleNonIdleCpuStat.size());
    for (std::size_t i = 0; i < idleNonIdleCpuStat.size(); ++i) {
        cpuLoad[i] = static_cast<double>(idleNonIdleCpuStat[i].first);
    }

    for (std::size_t i = 0; i < cpuLoad.size(); ++i) {
        cpuLoadSum[i] += cpuLoad[i];
    }
    ++samplesNumber;

    cpuLoadHistory.push_back(std::move(cpuLoad));
    if (cpuLoadHistory.size() > historySize) {
        cpuLoadHistory.pop_front();
    }
}

std::deque<std::vector<double>> CpuMonitor::getLastHistory() const {
    return cpuLoadHistory;
}

std::vector<double> CpuMonitor::getMeanCpuLoad() const {
    std::vector<double> meanCpuLoad;
    meanCpuLoad.reserve(cpuLoadSum.size());
    for (double coreLoad : cpuLoadSum) {
        meanCpuLoad.push_back(coreLoad / samplesNumber);
    }
    return meanCpuLoad;
}
#else
#include <regex>
#include <utility>
#include <fstream>

#include <unistd.h>

namespace {
std::vector<std::pair<unsigned long, unsigned long>> getIdleNonIdleCpuStat(std::size_t nCores) {
    std::vector<std::pair<unsigned long, unsigned long>> idleNonIdleCpuStat(nCores);
    std::ifstream proc_stat("/proc/stat");
    std::string line;
    std::smatch match;
    std::regex coreJiffies("^cpu(\\d+)\\s+"
        "(\\d+)\\s+" // user
        "(\\d+)\\s+" // nice
        "(\\d+)\\s+" // system
        "(\\d+)\\s+" // idle
        "(\\d+)\\s+" // iowait
        "(\\d+)\\s+" // irq
        "(\\d+)\\s+" // softirq
        "(\\d+)\\s+" // steal
        "(\\d+)\\s+" // guest
        "(\\d+)$");  // guest_nice
    while (std::getline(proc_stat, line))
    {
        if (std::regex_match(line, match, coreJiffies))
        {
            unsigned long idleInfo = stoul(match[5]) + stoul(match[6]),
                nonIdleInfo = stoul(match[2])
                    + stoul(match[3])
                    + stoul(match[4])
                    + stoul(match[7])
                    + stoul(match[8])
                    + stoul(match[9]), // it doesn't handle overflow of sum and overflows of /proc/stat values
                core_id = stoul(match[1]);
            idleNonIdleCpuStat[core_id].first = idleInfo;
            idleNonIdleCpuStat[core_id].second = nonIdleInfo;
        }
    }
    return idleNonIdleCpuStat;
}
}

CpuMonitor::CpuMonitor() :
    nCores{static_cast<std::size_t>(sysconf(_SC_NPROCESSORS_CONF))},
    lastEnabled{false},
    samplesNumber{0},
    historySize{0},
    cpuLoadSum(nCores, 0) {}

void CpuMonitor::enableHistory(std::size_t historySize) {
    if (historySize < 2) {
        disableHistory();
    } else {
        this->historySize = historySize;
        if (!lastEnabled) {
            prevIdleNonIdleCpuStat = getIdleNonIdleCpuStat(nCores);
        }
    }
}

bool CpuMonitor::isHistoryEnabled() const {
    return historySize > 1;
}

void CpuMonitor::disableHistory() {
    historySize = 1;
}

void CpuMonitor::enableLast() {
    lastEnabled = true;
    if (!isHistoryEnabled()) {
        historySize = 1;
        prevIdleNonIdleCpuStat = getIdleNonIdleCpuStat(nCores);
    }
}

bool CpuMonitor::isLastEnabled() const {
    return lastEnabled;
}

void CpuMonitor::disableLast() {
    lastEnabled = false;
}

void CpuMonitor::collectData() {
    // store CPU Utulization in range [0, 1]
    std::vector<std::pair<unsigned long, unsigned long>> idleNonIdleCpuStat = getIdleNonIdleCpuStat(nCores);
    std::vector<double> cpuLoad(idleNonIdleCpuStat.size());
    for (std::size_t i = 0; i < idleNonIdleCpuStat.size(); ++i) {
        unsigned long idleDiff = idleNonIdleCpuStat[i].first - prevIdleNonIdleCpuStat[i].first;
        unsigned long nonIdleDiff = idleNonIdleCpuStat[i].second - prevIdleNonIdleCpuStat[i].second;
        if (0 == idleDiff + nonIdleDiff) {
            if (cpuLoadHistory.empty()) {
                cpuLoad[i] = 0;
            } else {
                cpuLoad[i] = cpuLoadHistory.back()[i];
            }
        } else {
            cpuLoad[i] = static_cast<double>(nonIdleDiff) / (idleDiff + nonIdleDiff);
        }
    }
    prevIdleNonIdleCpuStat = std::move(idleNonIdleCpuStat);

    for (std::size_t i = 0; i < cpuLoad.size(); ++i) {
        cpuLoadSum[i] += cpuLoad[i];
    }
    ++samplesNumber;

    cpuLoadHistory.push_back(std::move(cpuLoad));
    if (cpuLoadHistory.size() > historySize) {
        cpuLoadHistory.pop_front();
    }
}

std::deque<std::vector<double>> CpuMonitor::getLastHistory() const {
    return cpuLoadHistory;
}

std::vector<double> CpuMonitor::getMeanCpuLoad() const {
    std::vector<double> meanCpuLoad;
    meanCpuLoad.reserve(cpuLoadSum.size());
    for (double coreLoad : cpuLoadSum) {
        meanCpuLoad.push_back(coreLoad / samplesNumber);
    }
    return meanCpuLoad;
}
#endif
