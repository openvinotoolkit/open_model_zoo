// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "metrics.h"

#include <iostream>
#include <limits>
#include <sstream>
#include <iomanip>

PerformanceMetrics::PerformanceMetrics(Duration timeWindow) {
    // defines the length of the timespan over which the 'current fps' value is calculated
    timeWindowSize = timeWindow;
}

void PerformanceMetrics::update(TimePoint lastRequestStartTime,
                                cv::Mat& frame,
                                cv::Point position,
                                double fontScale,
                                cv::Scalar color,
                                int thickness) {
    TimePoint currentTime = Clock::now();

    if (lastUpdateTime == TimePoint()) {
        lastUpdateTime = currentTime;
        return;
    }
    
    currentMovingStatistic.latency += currentTime - lastRequestStartTime;
    currentMovingStatistic.period = currentTime - lastUpdateTime;
    currentMovingStatistic.frameCount++;

    if (currentTime - lastUpdateTime > timeWindowSize) {
        lastMovingStatistic = currentMovingStatistic;
        totalStatistic.combine(lastMovingStatistic);
        currentMovingStatistic = Statistic();

        lastUpdateTime = currentTime;
    }

    // Draw performance stats over frame
    Metrics metrics = getLast();
    std::ostringstream out;
    if (!std::isnan(metrics.latency)) {
        out << "Latency: " << std::fixed << std::setprecision(1) << metrics.latency << " ms";
        putHighlightedText(frame, out.str(), position, cv::FONT_HERSHEY_COMPLEX, fontScale, color, thickness);
    }
    if (!std::isnan(metrics.fps)) {
        out.str("");
        out << "FPS: " << std::fixed << std::setprecision(1) << metrics.fps;
        putHighlightedText(frame, out.str(), {position.x, position.y + 30}, cv::FONT_HERSHEY_COMPLEX, fontScale, color,
                           thickness);
    }
}

PerformanceMetrics::Metrics PerformanceMetrics::getLast() const {
    Metrics metrics;

    metrics.latency = lastMovingStatistic.frameCount != 0
                      ? std::chrono::duration_cast<Us>(lastMovingStatistic.latency).count()
                        / (1.e3 * lastMovingStatistic.frameCount)
                      : std::numeric_limits<double>::signaling_NaN();
    metrics.fps = lastMovingStatistic.period != Duration::zero()
                  ? (1.e3 * lastMovingStatistic.frameCount)
                    / std::chrono::duration_cast<Ms>(lastMovingStatistic.period).count()
                  : std::numeric_limits<double>::signaling_NaN();
    
    return metrics;
}

PerformanceMetrics::Metrics PerformanceMetrics::getTotal() const {
    Metrics metrics;

    int frameCount = totalStatistic.frameCount + currentMovingStatistic.frameCount;
    if (frameCount != 0) {
        metrics.latency = std::chrono::duration_cast<Us>(
            totalStatistic.latency + currentMovingStatistic.latency).count() / (1.e3 * frameCount);
        metrics.fps = (1.e3 * frameCount) / std::chrono::duration_cast<Ms>(
                                                totalStatistic.period + currentMovingStatistic.period).count();
    } else {
        metrics.latency = std::numeric_limits<double>::signaling_NaN();
        metrics.fps = std::numeric_limits<double>::signaling_NaN();
    }

    return metrics;
}

void PerformanceMetrics::printTotal() const {
    Metrics metrics = getTotal();

    std::cout << "Latency: ";
    if (metrics.latency > std::numeric_limits<double>::epsilon()) {
        std::cout << std::fixed << std::setprecision(1) << metrics.latency << " ms";
    } else {
        std::cout << "N/A";
    }
    std::cout << std::endl;

    std::cout << "FPS: ";
    if (metrics.fps > std::numeric_limits<double>::epsilon()) {
        std::cout << std::fixed << std::setprecision(1) << metrics.fps;
    } else {
        std::cout << "N/A";
    }
    std::cout << std::endl;
}
