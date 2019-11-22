// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_monitor.h"
#include <algorithm>
#ifdef _WIN32
#include "query_wrapper.h"
#include <string>
#include <system_error>
#include <pdhmsg.h>
#include <windows.h>

namespace {
std::size_t getNCores() {
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
}
}

struct CpuMonitor::PerformanceCounter {
    PerformanceCounter(std::size_t nCores) : query{new QueryWrapper}, coreTimeCounters(nCores, 0) {}

    std::unique_ptr<QueryWrapper> query;
    std::vector<PDH_HCOUNTER> coreTimeCounters;
};

CpuMonitor::CpuMonitor() :
    nCores{getNCores()},
    lastEnabled{false},
    samplesNumber{0},
    historySize{0},
    cpuLoadSum(nCores, 0) {}

// PerformanceCounter is incomplete in header and destructor can't be defined implicitly
CpuMonitor::~CpuMonitor() = default;

void CpuMonitor::openQuery() {
    std::unique_ptr<CpuMonitor::PerformanceCounter> newPerformanceCounter{new CpuMonitor::PerformanceCounter{nCores}};

    PDH_STATUS status;
    for (std::size_t i = 0; i < nCores; ++i)
    {
        std::wstring fullCounterPath{L"\\Processor(" + std::to_wstring(i) + L")\\% Processor Time"};
        status = PdhAddCounterW(*newPerformanceCounter->query, fullCounterPath.c_str(), 0,
            &newPerformanceCounter->coreTimeCounters[i]);
        if (ERROR_SUCCESS != status)
        {
            throw std::system_error(status, std::system_category(), "PdhAddCounter() failed");
        }
        status = PdhSetCounterScaleFactor(newPerformanceCounter->coreTimeCounters[i], -2); // scale counter to [0, 1]
        if (ERROR_SUCCESS != status)
        {
            throw std::system_error(status, std::system_category(), "PdhSetCounterScaleFactor() failed");
        }
    }
    status = PdhCollectQueryData(*newPerformanceCounter->query);
    if (ERROR_SUCCESS != status)
    {
        throw std::system_error(status, std::system_category(), "PdhCollectQueryData() failed");
    }
    performanceCounter = std::move(newPerformanceCounter);
}

void CpuMonitor::closeQuery() {
    performanceCounter.reset();
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
    status = PdhCollectQueryData(*performanceCounter->query);
    if (ERROR_SUCCESS != status) {
        throw std::system_error(status, std::system_category(), "PdhCollectQueryData() failed");
    }

    PDH_FMT_COUNTERVALUE displayValue;
    std::vector<double> cpuLoad(performanceCounter->coreTimeCounters.size());
    for (std::size_t i = 0; i < performanceCounter->coreTimeCounters.size(); ++i) {
        status = PdhGetFormattedCounterValue(performanceCounter->coreTimeCounters[i], PDH_FMT_DOUBLE, NULL,
            &displayValue);
        if (ERROR_SUCCESS != status) {
            throw std::system_error(status, std::system_category(), "PdhGetFormattedCounterValue() failed");
        }
        if (PDH_CSTATUS_VALID_DATA != displayValue.CStatus && PDH_CSTATUS_NEW_DATA != displayValue.CStatus) {
            throw std::runtime_error("Error in counter data");
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
std::vector<unsigned long> getIdleCpuStat(std::size_t nCores) {
    std::vector<unsigned long> idleCpuStat(nCores);
    std::ifstream procStat("/proc/stat");
    std::string line;
    std::smatch match;
    std::regex coreJiffies("^cpu(\\d+)\\s+"
        "(\\d+)\\s+"
        "(\\d+)\\s+"
        "(\\d+)\\s+"
        "(\\d+)\\s+" // idle
        "(\\d+)"); // iowait

    while (std::getline(procStat, line)) {
        if (std::regex_search(line, match, coreJiffies)) {
            // it doesn't handle overflow of sum and overflows of /proc/stat values
            unsigned long idleInfo = stoul(match[5]) + stoul(match[6]),
                coreId = stoul(match[1]);
            if (nCores <= coreId) {
                throw std::runtime_error("The number of cores has changed");
            }
            idleCpuStat[coreId] = idleInfo;
        }
    }
    return idleCpuStat;
}

const long clockTicks = sysconf(_SC_CLK_TCK);
}

CpuMonitor::CpuMonitor() :
    nCores{static_cast<std::size_t>(sysconf(_SC_NPROCESSORS_CONF))},
    lastEnabled{false},
    samplesNumber{0},
    historySize{0},
    cpuLoadSum(nCores, 0) {}

void CpuMonitor::setHistorySize(std::size_t size) {
    if (0 == historySize && 0 != size) {
        prevIdleCpuStat = getIdleCpuStat(nCores);
        prevTimePoint = std::chrono::steady_clock::now();
    }
    historySize = size;
    std::size_t newSize = std::min(size, cpuLoadHistory.size());
    cpuLoadHistory.erase(cpuLoadHistory.begin(), cpuLoadHistory.end() - newSize);
}

std::size_t CpuMonitor::getHistorySize() const {
    return historySize;
}

void CpuMonitor::collectData() {
    std::vector<unsigned long> idleCpuStat = getIdleCpuStat(nCores);
    auto timePoint = std::chrono::steady_clock::now();
    // don't update data too frequently which may result in negative values for cpuLoad.
    // It may happen when collectData() is called just after setHistorySize().
    if (timePoint - prevTimePoint > std::chrono::milliseconds{100}) {
        std::vector<double> cpuLoad(idleCpuStat.size());
        for (std::size_t i = 0; i < idleCpuStat.size(); ++i) {
            double idleDiff = idleCpuStat[i] - prevIdleCpuStat[i];
            typedef std::chrono::duration<double, std::chrono::seconds::period> Sec;
            cpuLoad[i] = 1.0
                - idleDiff / clockTicks / std::chrono::duration_cast<Sec>(timePoint - prevTimePoint).count();
        }
        prevTimePoint = timePoint;
        prevIdleCpuStat = std::move(idleCpuStat);

        for (std::size_t i = 0; i < cpuLoad.size(); ++i) {
            cpuLoadSum[i] += cpuLoad[i];
        }
        ++samplesNumber;

        cpuLoadHistory.push_back(std::move(cpuLoad));
        if (cpuLoadHistory.size() > historySize) {
            cpuLoadHistory.pop_front();
        }
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
