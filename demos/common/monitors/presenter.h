// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_set>

#include <opencv2/imgproc.hpp>

#include "cpu_monitor.h"
#include "memory_monitor.h"

enum class MonitorType: int {CpuAverage, DistributionCpu, Memory};

class Presenter {
public:
    explicit Presenter(std::unordered_set<MonitorType> enabledMonitors = {},
        int yPos = 20,
        cv::Size graphSize = {150, 60},
        std::size_t historySize = 20);
    explicit Presenter(const std::string& keys,
        int yPos = 20,
        cv::Size graphSize = {150, 60},
        std::size_t historySize = 20);
    void addRemoveMonitor(MonitorType monitor);
    void addRemoveMonitor(int key); // handles c, d, m, h keys
    void drawGraphs(cv::Mat& frame);
    // std::map<MonitorType, Values> getMean() const TODO

    const int yPos;
    const cv::Size graphSize;
    const int graphPadding;
private:
    std::chrono::steady_clock::time_point prevTimeStamp;
    std::size_t historySize;
    CpuMonitor cpuMonitor;
    MemoryMonitor memoryMonitor;
    std::ostringstream strStream;
};
