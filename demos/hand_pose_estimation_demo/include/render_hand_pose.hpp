// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <opencv2/core/core.hpp>

#include "hand_pose.hpp"

namespace hand_pose_estimation {
    void renderhandPose(const std::vector<handPose>& poses, cv::Mat& image);
}  // namespace hand_pose_estimation
