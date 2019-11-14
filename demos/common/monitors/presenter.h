// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <ostream>
#include <set>

#include <opencv2/imgproc.hpp>

#include "cpu_monitor.h"
#include "memory_monitor.h"

enum class MonitorType{CpuAverage, DistributionCpu, Memory};

void meansToOstream(const std::map<MonitorType, std::vector<double>>, std::ostream& stream);

class Presenter {
public:
    explicit Presenter(std::set<MonitorType> enabledMonitors = {},
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
    std::map<MonitorType, std::vector<double>> getMeans() const;

    const int yPos;
    const cv::Size graphSize;
    const int graphPadding;
private:
    std::chrono::steady_clock::time_point prevTimeStamp;
    std::size_t historySize;
    CpuMonitor cpuMonitor;
    bool distributionCpuEnabled;
    MemoryMonitor memoryMonitor;
    std::ostringstream strStream;
};
