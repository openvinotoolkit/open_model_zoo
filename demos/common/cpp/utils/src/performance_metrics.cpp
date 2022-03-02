// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include "utils/performance_metrics.hpp"
#include "utils/slog.hpp"

// timeWindow defines the length of the timespan over which the 'current fps' value is calculated
PerformanceMetrics::PerformanceMetrics(Duration timeWindow)
    : timeWindowSize(timeWindow)
    , firstFrameProcessed(false)
{}

void PerformanceMetrics::update(TimePoint lastRequestStartTime,
    const cv::Mat& frame,
    cv::Point position,
    int fontFace,
    double fontScale,
    cv::Scalar color,
    int thickness,
    MetricTypes metricType) {
    update(lastRequestStartTime);
    paintMetrics(frame, position, fontFace, fontScale, color, thickness, metricType);
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

void PerformanceMetrics::paintMetrics(const cv::Mat& frame, cv::Point position, int fontFace,
    double fontScale, cv::Scalar color, int thickness, MetricTypes metricType) const {
    // Draw performance stats over frame
    Metrics metrics = getLast();

    std::ostringstream out;
    if (!std::isnan(metrics.latency) &&
        (metricType == PerformanceMetrics::MetricTypes::LATENCY || metricType == PerformanceMetrics::MetricTypes::ALL)) {
        out << "Latency: " << std::fixed << std::setprecision(1) << metrics.latency << " ms";
        putHighlightedText(frame, out.str(), position, fontFace, fontScale, color, thickness);
    }
    if (!std::isnan(metrics.fps) &&
        (metricType == PerformanceMetrics::MetricTypes::FPS || metricType == PerformanceMetrics::MetricTypes::ALL)) {
        out.str("");
        out << "FPS: " << std::fixed << std::setprecision(1) << metrics.fps;
        int offset = metricType == PerformanceMetrics::MetricTypes::ALL ? 30 : 0;
        putHighlightedText(frame, out.str(), {position.x, position.y + offset}, fontFace, fontScale, color, thickness);
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

void PerformanceMetrics::logTotal() const {
    Metrics metrics = getTotal();

    slog::info << "\tLatency: " << std::fixed << std::setprecision(1) << metrics.latency << " ms" << slog::endl;
    slog::info << "\tFPS: " << metrics.fps << slog::endl;
}

void logLatencyPerStage(double readLat, double preprocLat, double inferLat, double postprocLat, double renderLat) {
    slog::info << "\tDecoding:\t" << std::fixed << std::setprecision(1) <<
        readLat << " ms" << slog::endl;
    slog::info << "\tPreprocessing:\t" << preprocLat << " ms" << slog::endl;
    slog::info << "\tInference:\t" << inferLat << " ms" << slog::endl;
    slog::info << "\tPostprocessing:\t" << postprocLat << " ms" << slog::endl;
    slog::info << "\tRendering:\t" << renderLat << " ms" << slog::endl;
}
