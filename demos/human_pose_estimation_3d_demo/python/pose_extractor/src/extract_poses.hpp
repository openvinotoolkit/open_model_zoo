// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <opencv2/core/core.hpp>

#include "human_pose.hpp"

namespace human_pose_estimation {
std::vector<HumanPose> extractPoses(
        std::vector<cv::Mat>& heatMaps,
        std::vector<cv::Mat>& pafs,
        int upsampleRatio);
} // namespace human_pose_estimation
