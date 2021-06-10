// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/performance_metrics.hpp"

#include <limits>

// timeWindow defines the length of the timespan over which the 'current fps' value is calculated
PerformanceMetrics::PerformanceMetrics(Duration timeWindow)
    : timeWindowSize(timeWindow)
    , firstFrameProcessed(false)
{}

void PerformanceMetrics::update(TimePoint lastRequestStartTime,
    cv::Mat& frame,
    cv::Point position,
    int fontFace,
    double fontScale,
    cv::Scalar color,
    int thickness) {
    update(lastRequestStartTime);
    paintMetrics(frame, position, fontFace, fontScale, color, thickness);
}

void PerformanceMetrics::update(TimePoint lastRequestStartTime) {
    TimePoint currentTime = Clock::now();

    if (!firstFrameProcessed) {
        lastUpdateTime = lastRequestStartTime;
        firstFrameProcessed = true;
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
}

void PerformanceMetrics::paintMetrics(cv::Mat & frame, cv::Point position, int fontFace, double fontScale, cv::Scalar color, int thickness) const {
    // Draw performance stats over frame
    Metrics metrics = getLast();

    std::ostringstream out;
    if (!std::isnan(metrics.latency)) {
        out << "Latency: " << std::fixed << std::setprecision(1) << metrics.latency << " ms";
        putHighlightedText(frame, out.str(), position, fontFace, fontScale, color, thickness);
    }
    if (!std::isnan(metrics.fps)) {
        out.str("");
        out << "FPS: " << std::fixed << std::setprecision(1) << metrics.fps;
        putHighlightedText(frame, out.str(), {position.x, position.y + 30}, fontFace, fontScale, color, thickness);
    }
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

void PerformanceMetrics::printTotal() const {
    Metrics metrics = getTotal();

    std::ostringstream out;
    out << "Latency: " << std::fixed << std::setprecision(1) << metrics.latency << " ms\nFPS: " << metrics.fps << '\n';
    std::cout << out.str();
}
