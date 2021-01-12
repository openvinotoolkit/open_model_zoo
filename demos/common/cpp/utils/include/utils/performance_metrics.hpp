// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file for performance metrics calculation class
 * @file performance_metrics.hpp
 */

#pragma once

#include <chrono>
#include <iostream>
#include <sstream>
#include <iomanip>

#include "utils/ocv_common.hpp"

class PerformanceMetrics {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = Clock::duration;
    using Ms = std::chrono::duration<double, std::ratio<1, 1000>>;
    using Sec = std::chrono::duration<double, std::ratio<1, 1>>;

    struct Metrics {
        double latency;
        double fps;
    };

    PerformanceMetrics(Duration timeWindow = std::chrono::seconds(1));
    void update(TimePoint lastRequestStartTime,
                cv::Mat& frame,
                cv::Point position = {15, 30},
                int fontFace = cv::FONT_HERSHEY_COMPLEX,
                double fontScale = 0.75,
                cv::Scalar color = {200, 10, 10},
                int thickness = 2);
    void update(TimePoint lastRequestStartTime);

    /// Paints metrics over provided mat
    /// @param frame frame to paint over
    /// @param position left top corner of text block
    /// @param fontScale font scale
    /// @param color font color
    /// @param thickness font thickness
    void paintMetrics(cv::Mat& frame,
        cv::Point position = { 15, 30 },
        int fontFace = cv::FONT_HERSHEY_COMPLEX,
        double fontScale = 0.75,
        cv::Scalar color = { 200, 10, 10 },
        int thickness = 2) const;

    Metrics getLast() const;
    Metrics getTotal() const;
    void printTotal() const;

private:
    struct Statistic {
        Duration latency;
        Duration period;
        int frameCount;

        Statistic() {
            latency = Duration::zero();
            period = Duration::zero();
            frameCount = 0;
        }

        void combine(const Statistic& other) {
            latency += other.latency;
            period += other.period;
            frameCount += other.frameCount;
        }
    };

    Duration timeWindowSize;
    Statistic lastMovingStatistic;
    Statistic currentMovingStatistic;
    Statistic totalStatistic;
    TimePoint lastUpdateTime;
    bool firstFrameProcessed;
};
