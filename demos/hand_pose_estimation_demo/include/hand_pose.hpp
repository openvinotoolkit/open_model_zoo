// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <opencv2/core/core.hpp>

namespace hand_pose_estimation {
struct handPose {
    handPose(const std::vector<cv::Point2f>& keypoints = std::vector<cv::Point2f>(),
              const float& score = 0);

    std::vector<cv::Point2f> keypoints;
    float score;
};
}  // namespace hand_pose_estimation
