// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "metrics.h"

PerformanceMetrics::PerformanceMetrics(Duration timeWindow) {
    timeWindowSize = timeWindow;
    fps = 0;
    latency = 0;
    startTime = Clock::now();
    numFramesProcessed = 0;
}

void PerformanceMetrics::recalculate(TimePoint lastRequestStartTime) {
    auto currentTime = Clock::now();
    while (!measurements.empty()) {
        auto firstInWindow = measurements.front();
        if (currentTime - firstInWindow.timePoint > timeWindowSize) {
            latencySum -= firstInWindow.latency;
            measurements.pop_front();
        } else break;
    }

    auto lastRequestLatency = currentTime - lastRequestStartTime;
    measurements.push_back({lastRequestLatency, currentTime});
    latencySum += lastRequestLatency;
    latencyTotalSum += lastRequestLatency;
    latency = std::chrono::duration_cast<Ms>(latencySum).count() / measurements.size();

    auto spfSum = measurements.back().timePoint - measurements.front().timePoint;
    if (spfSum >= Ms(0)) {
        fps = 1000.0 * measurements.size() / std::chrono::duration_cast<Ms>(spfSum).count();
    }

    numFramesProcessed++;
}

double PerformanceMetrics::getTotalFps() const {
    return 1000.0 * numFramesProcessed / std::chrono::duration_cast<Ms>(stopTime - startTime).count();
}

double PerformanceMetrics::getTotalLatency() const {
    return std::chrono::duration_cast<Ms>(latencyTotalSum).count() / numFramesProcessed;
}

void PerformanceMetrics::stop() {
    stopTime = Clock::now();
}

void PerformanceMetrics::reinitialize() {
    measurements.clear();
    fps = 0;
    latency = 0;
    latencySum = Duration::zero();
    startTime = Clock::now();
    numFramesProcessed = 0;
    latencyTotalSum = Duration::zero();
}

bool PerformanceMetrics::hasStarted() const {
    return numFramesProcessed > 0;
}
