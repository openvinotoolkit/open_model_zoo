// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <opencv2/core/types.hpp>
#include "utils/ocv_common.hpp"

namespace gaze_estimation {
void gazeVectorToGazeAngles(const cv::Point3f& gazeVector, cv::Point2f& gazeAngles);
}  // namespace gaze_estimation
