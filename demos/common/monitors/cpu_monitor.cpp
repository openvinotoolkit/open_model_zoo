// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_monitor.h"
#include <algorithm>
#ifdef _WIN32
#include <windows.h>
#include <pdhmsg.h>
#include <string>
#include <system_error>

namespace {
std::size_t getNCores() {
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
}
}

QueryWrapper::QueryWrapper() {
    PDH_STATUS status = PdhOpenQuery(NULL, NULL, &query);
    if (ERROR_SUCCESS != status) {
        throw std::system_error(status, std::system_category(), "PdhOpenQuery() failed");
    }
}
QueryWrapper::~QueryWrapper() {
    PdhCloseQuery(query);
}

void QueryWrapper::closeQuery() {
    PDH_STATUS status = PdhCloseQuery(query);
    if (ERROR_SUCCESS != status) {
        throw std::logic_error("The query handle to close is not valid");
    }
}

QueryWrapper::operator PDH_HQUERY() const {
    return query;
}

CpuMonitor::CpuMonitor() :
    nCores{getNCores()},
    lastEnabled{false},
    samplesNumber{0},
    historySize{0},
    cpuLoadSum(nCores, 0) {}

void CpuMonitor::openQuery() {
    QueryWrapper newQueryWrapper;

    PDH_STATUS status;
    coreTimeCounters.resize(nCores);
    for (std::size_t i = 0; i < nCores; ++i)
    {
        std::wstring fullCounterPath{L"\\Processor(" + std::to_wstring(i) + L")\\% Processor Time"};
        status = PdhAddCounterW(newQueryWrapper, fullCounterPath.c_str(), 0, &coreTimeCounters[i]);
        if (ERROR_SUCCESS != status)
        {
            throw std::system_error(status, std::system_category(), "PdhAddCounter() failed");
        }
        status = PdhSetCounterScaleFactor(coreTimeCounters[i], -2); // scale counter to [0, 1]
        if (ERROR_SUCCESS != status)
        {
            throw std::system_error(status, std::system_category(), "PdhSetCounterScaleFactor() failed");
        }
    }
    status = PdhCollectQueryData(newQueryWrapper);
    if (ERROR_SUCCESS != status)
    {
        throw std::system_error(status, std::system_category(), "PdhCollectQueryData() failed");
    }
    QueryWrapper dummy;
    queryWrapper = newQueryWrapper;
    newQueryWrapper = dummy;
}

void CpuMonitor::closeQuery() {
    queryWrapper.closeQuery();
    coreTimeCounters.clear();
}

void CpuMonitor::setHistorySize(std::size_t size) {
    if (0 == historySize && 0 != size) {
        openQuery();
    } else if (0 != historySize && 0 == size) {
        closeQuery();
    }
    historySize = size;
    std::size_t newSize = std::min(size, cpuLoadHistory.size());
    cpuLoadHistory.erase(cpuLoadHistory.begin(), cpuLoadHistory.end() - newSize);
}

std::size_t CpuMonitor::getHistorySize() const {
    return historySize;
}

void CpuMonitor::collectData() {
    PDH_STATUS status;
    status = PdhCollectQueryData(queryWrapper);
    if (ERROR_SUCCESS != status) {
        throw std::system_error(status, std::system_category(), "PdhCollectQueryData() failed");
    }

    PDH_FMT_COUNTERVALUE displayValue;
    std::vector<double> cpuLoad(coreTimeCounters.size());
    for (std::size_t i = 0; i < coreTimeCounters.size(); i++) {
        status = PdhGetFormattedCounterValue(coreTimeCounters[i], PDH_FMT_DOUBLE, NULL, &displayValue);
        if (ERROR_SUCCESS != status) {
            throw std::system_error(status, std::system_category(), "PdhGetFormattedCounterValue() failed");
        }
        if (PDH_CSTATUS_VALID_DATA != displayValue.CStatus && PDH_CSTATUS_NEW_DATA != displayValue.CStatus) {
            throw std::system_error(status, std::system_category(), "Error in counter data");
        }

        cpuLoad[i] = displayValue.doubleValue;
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
    std::ifstream procStat("/proc/stat");
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
        "(\\d+)");  // guest_nice
    while (std::getline(procStat, line))
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
                coreId = stoul(match[1]);
            if (nCores <= coreId) {
                throw std::runtime_error("The number of cores has changed");
            }
            idleNonIdleCpuStat[coreId].first = idleInfo;
            idleNonIdleCpuStat[coreId].second = nonIdleInfo;
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

void CpuMonitor::setHistorySize(std::size_t size) {
    if (0 == historySize && 0 != size) {
        prevIdleNonIdleCpuStat = getIdleNonIdleCpuStat(nCores);
    }
    historySize = size;
    std::size_t newSize = std::min(size, cpuLoadHistory.size());
    cpuLoadHistory.erase(cpuLoadHistory.begin(), cpuLoadHistory.end() - newSize);
}

std::size_t CpuMonitor::getHistorySize() const {
    return historySize;
}

void CpuMonitor::collectData() {
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
