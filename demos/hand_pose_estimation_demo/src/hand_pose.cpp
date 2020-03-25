// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "hand_pose.hpp"

namespace hand_pose_estimation {
handPose::handPose(const std::vector<cv::Point2f>& keypoints,
                     const float& score)
    : keypoints(keypoints),
      score(score) {}
}  // namespace hand_pose_estimation
