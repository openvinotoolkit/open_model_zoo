// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <vector>

#include <samples/common.hpp>

#include "human_pose_estimator.hpp"
#include "send_human_pose.hpp"


namespace human_pose_estimation {

void sendHumanPose(const std::vector<HumanPose>& poses) {
    if (!poses.empty()) {
        std::time_t result = std::time(nullptr);
        std::cout << std::asctime(std::localtime(&result)) << std::endl;
     }

    for (HumanPose const& pose : poses) {
        std::stringstream rawPose;
        rawPose << std::fixed << std::setprecision(0);
        for (auto const& keypoint : pose.keypoints) {
            rawPose << keypoint.x << "," << keypoint.y << " ";
        }
        rawPose << pose.score;
        std::cout << rawPose.str() << std::endl;
    }
}
}  // namespace human_pose_estimation
