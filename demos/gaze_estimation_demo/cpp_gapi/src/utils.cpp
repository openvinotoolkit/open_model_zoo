// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define _USE_MATH_DEFINES

#include "utils.hpp"

#include <cmath>

#include <opencv2/core.hpp>

namespace gaze_estimation {
void gazeVectorToGazeAngles(const cv::Point3f& gazeVector, cv::Point2f& gazeAngles) {
    auto r = cv::norm(gazeVector);

    double v0 = static_cast<double>(gazeVector.x);
    double v1 = static_cast<double>(gazeVector.y);
    double v2 = static_cast<double>(gazeVector.z);

    gazeAngles.x = static_cast<float>(180.0 / M_PI * (M_PI_2 + std::atan2(v2, v0)));
    gazeAngles.y = static_cast<float>(180.0 / M_PI * (M_PI_2 - std::acos(v1 / r)));
}
}  // namespace gaze_estimation
