// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <opencv2/core/core.hpp>

namespace human_pose_estimation {
struct HumanPose {
    HumanPose(const std::vector<cv::Point3f>& keypoints = std::vector<cv::Point3f>(),
              const float& score = 0);

    std::vector<cv::Point3f> keypoints;
    float score;
};
} // namespace human_pose_estimation

