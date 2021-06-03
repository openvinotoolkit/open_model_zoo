// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define _USE_MATH_DEFINES

#include "utils.hpp"

namespace gaze_estimation {
void gazeVectorToGazeAngles(const cv::Point3f& gazeVector, cv::Point2f& gazeAngles) {
    auto r = cv::norm(gazeVector);

    double v0 = static_cast<double>(gazeVector.x);
    double v1 = static_cast<double>(gazeVector.y);
    double v2 = static_cast<double>(gazeVector.z);

    gazeAngles.x = static_cast<float>(180.0 / M_PI * (M_PI_2 + std::atan2(v2, v0)));
    gazeAngles.y = static_cast<float>(180.0 / M_PI * (M_PI_2 - std::acos(v1 / r)));
}

void putTimingInfoOnFrame(cv::Mat& image, double overallTime) {
    auto frameHeight = image.rows;
    double fontScale = 1.6 * frameHeight / 640;
    auto fontColor = cv::Scalar(0, 0, 255);
    int thickness = 2;

    double overallFPS = 1000. / overallTime;

    const auto format = cv::format("Overall FPS: %0.0f", overallFPS);
    cv::putText(image,
                format,
                cv::Point(10, static_cast<int>(30 * fontScale / 1.6)), cv::FONT_HERSHEY_PLAIN, fontScale, fontColor, thickness);
}
}  // namespace gaze_estimation
