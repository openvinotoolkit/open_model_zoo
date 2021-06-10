// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>

namespace gaze_estimation {
void gazeVectorToGazeAngles(const cv::Point3f& gazeVector, cv::Point2f& gazeAngles);

void putTimingInfoOnFrame(cv::Mat& image, double overallTime);
}  // namespace gaze_estimation
