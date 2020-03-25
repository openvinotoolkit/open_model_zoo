// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_monitor.h"

struct MemState {
    double memTotal, usedMem, usedSwap;
};

#ifdef _WIN32
#include "query_wrapper.h"
#include <algorithm>
#define PSAPI_VERSION 2
#include <system_error>
#include <windows.h>
#include <pdhmsg.h>
#include <psapi.h>

namespace {
double getMemTotal() {
    PERFORMANCE_INFORMATION performanceInformation;
    if (!GetPerformanceInfo(&performanceInformation, sizeof(performanceInformation))) {
        throw std::runtime_error("GetPerformanceInfo() failed");
    }
    return static_cast<double>(performanceInformation.PhysicalTotal * performanceInformation.PageSize)
        / (1024 * 1024 * 1024);
}
}

class MemoryMonitor::PerformanceCounter {
public:
    PerformanceCounter() {
        PDH_STATUS status = PdhAddCounterW(query, L"\\Paging File(_Total)\\% Usage", 0, &pagingFileUsageCounter);
        if (ERROR_SUCCESS != status) {
            throw std::system_error(status, std::system_category(), "PdhAddCounterW() failed");
        }
        status = PdhSetCounterScaleFactor(pagingFileUsageCounter, -2); // scale counter to [0, 1]
        if (ERROR_SUCCESS != status) {
            throw std::system_error(status, std::system_category(), "PdhSetCounterScaleFactor() failed");
        }
    }

    MemState getMemState() {
        PERFORMANCE_INFORMATION performanceInformation;
        if (!GetPerformanceInfo(&performanceInformation, sizeof(performanceInformation))) {
            throw std::runtime_error("GetPerformanceInfo() failed");
        }

        PDH_STATUS status;
        status = PdhCollectQueryData(query);
        if (ERROR_SUCCESS != status) {
            throw std::system_error(status, std::system_category(), "PdhCollectQueryData() failed");
        }
        PDH_FMT_COUNTERVALUE displayValue;
        status = PdhGetFormattedCounterValue(pagingFileUsageCounter, PDH_FMT_DOUBLE, NULL, &displayValue);
        if (ERROR_SUCCESS != status) {
            throw std::system_error(status, std::system_category(), "PdhGetFormattedCounterValue() failed");
        }
        if (PDH_CSTATUS_VALID_DATA != displayValue.CStatus && PDH_CSTATUS_NEW_DATA != displayValue.CStatus) {
            throw std::runtime_error("Error in counter data");
        }

        double pagingFilesSize = static_cast<double>(
            (performanceInformation.CommitLimit - performanceInformation.PhysicalTotal)
            * performanceInformation.PageSize) / (1024 * 1024 * 1024);
        return {static_cast<double>(performanceInformation.PhysicalTotal * performanceInformation.PageSize)
                / (1024 * 1024 * 1024),
            static_cast<double>(
                (performanceInformation.PhysicalTotal - performanceInformation.PhysicalAvailable)
                * performanceInformation.PageSize) / (1024 * 1024 * 1024),
            pagingFilesSize * displayValue.doubleValue};
    }
private:
    QueryWrapper query;
    PDH_HCOUNTER pagingFileUsageCounter;
};

#elif __linux__
#include <fstream>
#include <utility>
#include <vector>
#include <regex>

namespace {
std::pair<std::pair<double, double>, std::pair<double, double>> getAvailableMemSwapTotalMemSwap() {
    double memAvailable = 0, swapFree = 0, memTotal = 0, swapTotal = 0;
    std::regex memRegex("^(.+):\\s+(\\d+) kB$");
    std::string line;
    std::smatch match;
    std::ifstream meminfo("/proc/meminfo");
    while (std::getline(meminfo, line)) {
        if (std::regex_match(line, match, memRegex)) {
            if ("MemAvailable" == match[1]) {
                memAvailable = stod(match[2]) / (1024 * 1024);
            } else if ("SwapFree" == match[1]) {
                swapFree = stod(match[2]) / (1024 * 1024);
            } else if ("MemTotal" == match[1]) {
                memTotal = stod(match[2]) / (1024 * 1024);
            } else if ("SwapTotal" == match[1]) {
                swapTotal = stod(match[2]) / (1024 * 1024);
            }
        }
    }
    if (0 == memTotal) {
        throw std::runtime_error("Can't get MemTotal");
    }
    return {{memAvailable, swapFree}, {memTotal, swapTotal}};
}

double getMemTotal() {
    return getAvailableMemSwapTotalMemSwap().second.first;
}
}

class MemoryMonitor::PerformanceCounter {
public:
    MemState getMemState() {
        std::pair<std::pair<double, double>, std::pair<double, double>> availableMemSwapTotalMemSwap
            = getAvailableMemSwapTotalMemSwap();
        double memTotal = availableMemSwapTotalMemSwap.second.first;
        double swapTotal = availableMemSwapTotalMemSwap.second.second;
        return {memTotal, memTotal - availableMemSwapTotalMemSwap.first.first, swapTotal - availableMemSwapTotalMemSwap.first.second};
    }
};

#else
// not implemented
namespace {
double getMemTotal() {return 0.0;}
}

class MemoryMonitor::PerformanceCounter {
public:
    MemState getMemState() {return {0.0, 0.0, 0.0};}
};
#endif

MemoryMonitor::MemoryMonitor() :
    samplesNumber{0},
    historySize{0},
    memSum{0.0},
    swapSum{0.0},
    maxMem{0.0},
    maxSwap{0.0},
    memTotal{0.0},
    maxMemTotal{0.0} {}

// PerformanceCounter is incomplete in header and destructor can't be defined implicitly
MemoryMonitor::~MemoryMonitor() = default;

void MemoryMonitor::setHistorySize(std::size_t size) {
    if (0 == historySize && 0 != size) {
        performanceCounter.reset(new MemoryMonitor::PerformanceCounter);
        // memTotal is not initialized in constructor because for linux its initialization involves constructing
        // std::regex which is unimplemented and throws an exception for gcc 4.8.5 (default for CentOS 7.4).
        // Delaying initialization triggers the error only when the monitor is used
        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=53631
        memTotal = ::getMemTotal();
    } else if (0 != historySize && 0 == size) {
        performanceCounter.reset();
    }
    historySize = size;
    std::size_t newSize = std::min(size, memSwapUsageHistory.size());
    memSwapUsageHistory.erase(memSwapUsageHistory.begin(), memSwapUsageHistory.end() - newSize);
}

void MemoryMonitor::collectData() {
    MemState memState = performanceCounter->getMemState();
    maxMemTotal = std::max(maxMemTotal, memState.memTotal);
    memSum += memState.usedMem;
    swapSum += memState.usedSwap;
    ++samplesNumber;
    maxMem = std::max(maxMem, memState.usedMem);
    maxSwap = std::max(maxSwap, memState.usedSwap);

    memSwapUsageHistory.emplace_back(memState.usedMem, memState.usedSwap);
    if (memSwapUsageHistory.size() > historySize) {
        memSwapUsageHistory.pop_front();
    }
}

std::size_t MemoryMonitor::getHistorySize() const {
    return historySize;
}

std::deque<std::pair<double, double>> MemoryMonitor::getLastHistory() const {
    return memSwapUsageHistory;
}

double MemoryMonitor::getMeanMem() const {
    return memSum / samplesNumber;
}

double MemoryMonitor::getMeanSwap() const {
    return swapSum / samplesNumber;
}

double MemoryMonitor::getMaxMem() const {
    return maxMem;
}

double MemoryMonitor::getMaxSwap() const {
    return maxSwap;
}

double MemoryMonitor::getMemTotal() const {
    return memTotal;
}

double MemoryMonitor::getMaxMemTotal() const {
    return maxMemTotal;
}
