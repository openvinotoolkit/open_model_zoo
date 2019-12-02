// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <opencv2/core/core.hpp>

#include "human_pose.hpp"
#include  <mqtt.h>

namespace human_pose_estimation {
    void sendHumanPose(int frame, mqtt * publisher, const std::vector<HumanPose>& poses);
}  // namespace human_pose_estimation
