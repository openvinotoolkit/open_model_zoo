// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>

#include <opencv2/imgproc.hpp>

/**
 * @brief Puts text message on the frame, highlights the text with a white border to make it distinguishable from
 *        the background.
 * @param frame - frame to put the text on.
 * @param message - text of the message.
 * @param position - bottom-left corner of the text string in the image.
 * @param fontFace - font type.
 * @param fontScale - font scale factor that is multiplied by the font-specific base size.
 * @param color - text color.
 * @param thickness - thickness of the lines used to draw a text.
 */
inline void putHighlightedText(cv::Mat& frame,
                               const std::string& message,
                               cv::Point position,
                               int fontFace,
                               double fontScale,
                               cv::Scalar color,
                               int thickness) {
    cv::putText(frame, message, position, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness + 1);
    cv::putText(frame, message, position, fontFace, fontScale, color, thickness);
}

class PerformanceMetrics {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = Clock::duration;
    using Ms = std::chrono::milliseconds;
    using Sec = std::chrono::seconds;

    struct Metrics {
        double latency;
        double fps;
    };

    PerformanceMetrics(Duration timeWindow = Sec(1));
    void update(TimePoint lastRequestStartTime,
                cv::Mat& frame,
                cv::Point position = {15, 30},
                double fontScale = 0.75,
                cv::Scalar color = {200, 10, 10},
                int thickness = 2);
    Metrics getLast();
    Metrics getTotal();
    void printTotal();

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
};
