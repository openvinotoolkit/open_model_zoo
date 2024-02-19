// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdio>
#include <string>

#define _USE_MATH_DEFINES
#include <cmath>

#include <utility>
#include <memory>
#include <map>
#include <vector>
#include <set>

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
}  // namespace gaze_estimation
