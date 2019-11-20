// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_monitor.h"

#ifdef _WIN32
#include <algorithm>

#define PSAPI_VERSION 2
#include <windows.h>
#include <pdhmsg.h>
#include <psapi.h>

MemoryMonitor::MemoryMonitor() :
        enabled{false},
        samplesNumber{0},
        historySize{0},
        memSum{0.0},
        swapSum{0.0},
        maxMem{0.0},
        maxSwap{0.0} {
    PERFORMANCE_INFORMATION performanceInformation;
    if (!GetPerformanceInfo(&performanceInformation, sizeof(performanceInformation))) {
        throw std::runtime_error("GetPerformanceInfo() failed");
    }
    maxMemTotal = memTotal = static_cast<double>(performanceInformation.PhysicalTotal * performanceInformation.PageSize)
        / (1024 * 1024 * 1024);
}

void MemoryMonitor::openQuery() {
    std::unique_ptr<QueryWrapper> newQuery{new QueryWrapper};

    PDH_STATUS status = PdhAddCounterW(*newQuery, L"\\Paging File(_Total)\\% Usage", 0, &pagingFileUsageCounter);
    if (ERROR_SUCCESS != status)
    {
        throw std::system_error(status, std::system_category(), "PdhSetCounterScaleFactor() failed");
    }
    status = PdhSetCounterScaleFactor(pagingFileUsageCounter, -2); // scale counter to [0, 1]
    if (ERROR_SUCCESS != status)
    {
        throw std::system_error(status, std::system_category(), "PdhSetCounterScaleFactor() failed");
    }
    query = std::move(newQuery);
}

void MemoryMonitor::closeQuery() {
    query.reset();
}

void MemoryMonitor::setHistorySize(std::size_t size) {
    if (0 == historySize && 0 != size) {
        openQuery();
    } else if (0 != historySize && 0 == size) {
        closeQuery();
    }
    historySize = size;
    std::size_t newSize = std::min(size, memSwapUsageHistory.size());
    memSwapUsageHistory.erase(memSwapUsageHistory.begin(), memSwapUsageHistory.end() - newSize);
}

std::size_t MemoryMonitor::getHistorySize() const {
    return historySize;
}

void MemoryMonitor::collectData() {
    PERFORMANCE_INFORMATION performanceInformation;
    if (!GetPerformanceInfo(&performanceInformation, sizeof(performanceInformation))) {
        throw std::runtime_error("GetPerformanceInfo() failed");
    }

    PDH_STATUS status;
    status = PdhCollectQueryData(*query);
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

    double usedMem = static_cast<double>(
        (performanceInformation.PhysicalTotal - performanceInformation.PhysicalAvailable)
        * performanceInformation.PageSize) / (1024 * 1024 * 1024);
    memTotal = static_cast<double>(performanceInformation.PhysicalTotal * performanceInformation.PageSize)
        / (1024 * 1024 * 1024);
    maxMemTotal = std::max(maxMemTotal, memTotal);
    double pagingFilesSize = static_cast<double>(
        (performanceInformation.CommitLimit - performanceInformation.PhysicalTotal)
        * performanceInformation.PageSize) / (1024 * 1024 * 1024);
    double usedSwap = pagingFilesSize * displayValue.doubleValue;

    memSum += usedMem;
    swapSum += usedSwap;
    ++samplesNumber;
    maxMem = std::max(maxMem, usedMem);
    maxSwap = std::max(maxSwap, usedSwap);

    memSwapUsageHistory.emplace_back(usedMem, usedSwap);
    if (memSwapUsageHistory.size() > historySize) {
        memSwapUsageHistory.pop_front();
    }
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

#else
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
}

MemoryMonitor::MemoryMonitor() :
        enabled{false},
        samplesNumber{0},
        historySize{0},
        memSum{0.0},
        swapSum{0.0},
        maxMem{0.0},
        maxSwap{0.0} {
    maxMemTotal = memTotal = getAvailableMemSwapTotalMemSwap().second.first;
}

void MemoryMonitor::setHistorySize(std::size_t size) {
    historySize = size;
    std::size_t newSize = std::min(size, memSwapUsageHistory.size());
    memSwapUsageHistory.erase(memSwapUsageHistory.begin(), memSwapUsageHistory.end() - newSize);
}

std::size_t MemoryMonitor::getHistorySize() const {
    return historySize;
}

void MemoryMonitor::collectData() {
    std::pair<std::pair<double, double>, std::pair<double, double>> availableMemSwapTotalMemSwap
        = getAvailableMemSwapTotalMemSwap();
    memTotal = availableMemSwapTotalMemSwap.second.first;
    double swapTotal = availableMemSwapTotalMemSwap.second.second;
    maxMemTotal = std::max(maxMemTotal, memTotal);
    double usedMem = memTotal - availableMemSwapTotalMemSwap.first.first;
    double usedSwap = swapTotal - availableMemSwapTotalMemSwap.first.second;

    memSum += usedMem;
    swapSum += usedSwap;
    ++samplesNumber;
    maxMem = std::max(maxMem, usedMem);
    maxSwap = std::max(maxSwap, usedSwap);

    memSwapUsageHistory.emplace_back(usedMem, usedSwap);
    if (memSwapUsageHistory.size() > historySize) {
        memSwapUsageHistory.pop_front();
    }
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
#endif
