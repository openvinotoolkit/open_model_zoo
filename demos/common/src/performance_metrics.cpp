// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "samples/performance_metrics.hpp"

#include <limits>

// timeWindow defines the length of the timespan over which the 'current fps' value is calculated
PerformanceMetrics::PerformanceMetrics(Duration timeWindow) : timeWindowSize(timeWindow) {}

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
    
    getLatencyMessage(out, metrics.latency, !std::isnan(metrics.latency));
    putHighlightedText(frame, out.str(), position, cv::FONT_HERSHEY_COMPLEX, fontScale, color, thickness);

    getFpsMessage(out, metrics.fps, !std::isnan(metrics.fps));
    putHighlightedText(frame, out.str(), {position.x, position.y + 30}, cv::FONT_HERSHEY_COMPLEX, fontScale, color,
                       thickness);
}

PerformanceMetrics::Metrics PerformanceMetrics::getLast() const {
    Metrics metrics;

    metrics.latency = lastMovingStatistic.frameCount != 0
                      ? std::chrono::duration_cast<Ms>(lastMovingStatistic.latency).count()
                        / lastMovingStatistic.frameCount
                      : std::numeric_limits<double>::signaling_NaN();
    metrics.fps = lastMovingStatistic.period != Duration::zero()
                  ? lastMovingStatistic.frameCount
                    / std::chrono::duration_cast<Sec>(lastMovingStatistic.period).count()
                  : std::numeric_limits<double>::signaling_NaN();
    
    return metrics;
}

PerformanceMetrics::Metrics PerformanceMetrics::getTotal() const {
    Metrics metrics;

    int frameCount = totalStatistic.frameCount + currentMovingStatistic.frameCount;
    if (frameCount != 0) {
        metrics.latency = std::chrono::duration_cast<Ms>(
            totalStatistic.latency + currentMovingStatistic.latency).count() / frameCount;
        metrics.fps = frameCount / std::chrono::duration_cast<Sec>(
                                       totalStatistic.period + currentMovingStatistic.period).count();
    } else {
        metrics.latency = std::numeric_limits<double>::signaling_NaN();
        metrics.fps = std::numeric_limits<double>::signaling_NaN();
    }

    return metrics;
}

void PerformanceMetrics::printTotal() {
    Metrics metrics = getTotal();
    std::ostringstream out;

    getLatencyMessage(out, metrics.latency, bool(metrics.latency > std::numeric_limits<double>::epsilon()));
    std::cout << out.str() << std::endl;

    getFpsMessage(out, metrics.fps, bool(metrics.fps > std::numeric_limits<double>::epsilon()));
    std::cout << out.str() << std::endl;
}

void PerformanceMetrics::getLatencyMessage(std::ostringstream& out, double value, bool isAvailable) {
    out.str("");

    out << "Latency: ";
    if (isAvailable) {
        out << std::fixed << std::setprecision(1) << value << " ms";
    } else {
        out << "N/A";
    }
}

void PerformanceMetrics::getFpsMessage(std::ostringstream& out, double value, bool isAvailable) {
    out.str("");

    out << "FPS: ";
    if (isAvailable) {
        out << std::fixed << std::setprecision(1) << value;
    } else {
        out << "N/A";
    }
}
