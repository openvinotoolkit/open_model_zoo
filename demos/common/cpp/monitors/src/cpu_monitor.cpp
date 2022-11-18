// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "monitors/cpu_monitor.h"

#include <algorithm>
#ifdef _WIN32
#include "query_wrapper.h"
#include <string>
#include <system_error>
#include <PdhMsg.h>
#include <Windows.h>

namespace {
const std::size_t nCores = []() {
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        return sysinfo.dwNumberOfProcessors;
    }();
}

class CpuMonitor::PerformanceCounter {
public:
    PerformanceCounter() : coreTimeCounters(nCores) {
        PDH_STATUS status;
        for (std::size_t i = 0; i < nCores; ++i) {
            std::wstring fullCounterPath{L"\\Processor(" + std::to_wstring(i) + L")\\% Processor Time"};
            status = PdhAddCounterW(query, fullCounterPath.c_str(), 0, &coreTimeCounters[i]);
            if (ERROR_SUCCESS != status) {
                throw std::system_error(status, std::system_category(), "PdhAddCounterW() failed");
            }
            status = PdhSetCounterScaleFactor(coreTimeCounters[i], -2); // scale counter to [0, 1]
            if (ERROR_SUCCESS != status) {
                throw std::system_error(status, std::system_category(), "PdhSetCounterScaleFactor() failed");
            }
        }
        status = PdhCollectQueryData(query);
        if (ERROR_SUCCESS != status) {
            throw std::system_error(status, std::system_category(), "PdhCollectQueryData() failed");
        }
    }

    std::vector<double> getCpuLoad() {
        PDH_STATUS status;
        status = PdhCollectQueryData(query);
        if (ERROR_SUCCESS != status) {
            throw std::system_error(status, std::system_category(), "PdhCollectQueryData() failed");
        }

        PDH_FMT_COUNTERVALUE displayValue;
        std::vector<double> cpuLoad(coreTimeCounters.size());
        for (std::size_t i = 0; i < coreTimeCounters.size(); ++i) {
            status = PdhGetFormattedCounterValue(coreTimeCounters[i], PDH_FMT_DOUBLE, NULL,
                &displayValue);
            switch (status) {
                case ERROR_SUCCESS: break;
                // PdhGetFormattedCounterValue() can sometimes return PDH_CALC_NEGATIVE_DENOMINATOR for some reason
                case PDH_CALC_NEGATIVE_DENOMINATOR: return {};
                default:
                    throw std::system_error(status, std::system_category(), "PdhGetFormattedCounterValue() failed");
            }
            if (PDH_CSTATUS_VALID_DATA != displayValue.CStatus && PDH_CSTATUS_NEW_DATA != displayValue.CStatus) {
                throw std::runtime_error("Error in counter data");
            }

            cpuLoad[i] = displayValue.doubleValue;
        }
        return cpuLoad;
    }

private:
    QueryWrapper query;
    std::vector<PDH_HCOUNTER> coreTimeCounters;
};

#elif __linux__
#include <chrono>
#include <regex>
#include <utility>
#include <fstream>
#include <unistd.h>

namespace {
const long clockTicks = sysconf(_SC_CLK_TCK);

const std::size_t nCores = sysconf(_SC_NPROCESSORS_CONF);

std::vector<unsigned long> getIdleCpuStat() {
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
}

class CpuMonitor::PerformanceCounter {
public:
    PerformanceCounter() : prevIdleCpuStat{getIdleCpuStat()}, prevTimePoint{std::chrono::steady_clock::now()} {}

    std::vector<double> getCpuLoad() {
        std::vector<unsigned long> idleCpuStat = getIdleCpuStat();
        auto timePoint = std::chrono::steady_clock::now();
        // don't update data too frequently which may result in negative values for cpuLoad.
        // It may happen when collectData() is called just after setHistorySize().
        if (timePoint - prevTimePoint > std::chrono::milliseconds{100}) {
            std::vector<double> cpuLoad(nCores);
            for (std::size_t i = 0; i < idleCpuStat.size(); ++i) {
                double idleDiff = idleCpuStat[i] - prevIdleCpuStat[i];
                typedef std::chrono::duration<double, std::chrono::seconds::period> Sec;
                cpuLoad[i] = 1.0
                    - idleDiff / clockTicks / std::chrono::duration_cast<Sec>(timePoint - prevTimePoint).count();
            }
            prevIdleCpuStat = std::move(idleCpuStat);
            prevTimePoint = timePoint;
            return cpuLoad;
        }
        return {};
    }
private:
    std::vector<unsigned long> prevIdleCpuStat;
    std::chrono::steady_clock::time_point prevTimePoint;
};

#else
// not implemented
namespace {
const std::size_t nCores{0};
}

class CpuMonitor::PerformanceCounter {
public:
    std::vector<double> getCpuLoad() {return {};};
};
#endif

CpuMonitor::CpuMonitor() :
    samplesNumber{0},
    historySize{0},
    cpuLoadSum(nCores, 0) {}

// PerformanceCounter is incomplete in header and destructor can't be defined implicitly
CpuMonitor::~CpuMonitor() = default;

void CpuMonitor::setHistorySize(std::size_t size) {
    if (0 == historySize && 0 != size) {
        performanceCounter.reset(new PerformanceCounter);
    } else if (0 != historySize && 0 == size) {
        performanceCounter.reset();
    }
    historySize = size;
    std::ptrdiff_t newSize = static_cast<std::ptrdiff_t>(std::min(size, cpuLoadHistory.size()));
    cpuLoadHistory.erase(cpuLoadHistory.begin(), cpuLoadHistory.end() - newSize);
}

void CpuMonitor::collectData() {
    std::vector<double> cpuLoad = performanceCounter->getCpuLoad();

    if (!cpuLoad.empty()) {
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

std::size_t CpuMonitor::getHistorySize() const {
    return historySize;
}

std::deque<std::vector<double>> CpuMonitor::getLastHistory() const {
    return cpuLoadHistory;
}

std::vector<double> CpuMonitor::getMeanCpuLoad() const {
    std::vector<double> meanCpuLoad;
    meanCpuLoad.reserve(cpuLoadSum.size());
    for (double coreLoad : cpuLoadSum) {
        meanCpuLoad.push_back(samplesNumber ? coreLoad / samplesNumber : 0);
    }
    return meanCpuLoad;
}
