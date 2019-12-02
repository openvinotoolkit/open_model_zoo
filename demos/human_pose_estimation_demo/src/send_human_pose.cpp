// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <vector>
#include <ctime>

#include <samples/common.hpp>

#include "human_pose_estimator.hpp"
#include "send_human_pose.hpp"


namespace human_pose_estimation {

char POSE_COCO_BODY_PARTS[][18] = { "Nose", "Neck", "RShoulder", "RElbow", "RWrist",  "LShoulder",
                                   "LElbow", "LWrist", "RHip", "RKnee", "RAnkle", "LHip",
                                    "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "Bkg"};

void sendHumanPose(int frame, mqtt *publisher, const std::vector<HumanPose>& poses) {

    std::time_t now = std::time(nullptr);
    if (!poses.empty()) {
        std::cout << std::asctime(std::localtime(&now)) << std::endl;
     }
    int pose_id= 0;
    for (HumanPose const& pose : poses) {
        std::stringstream rawPose;
        std::tm * ptm = std::localtime(&now);
        char timestamp[20];
        // Format: 15/06/2009-20:20:00
        std::strftime(timestamp, 20, "%d/%m/%Y-%H:%M:%S", ptm);
        rawPose << std::fixed << std::setprecision(0);
        // timestamp frame pose_id keypoints score
        rawPose << timestamp << " " << std::to_string(frame) << " " << std::to_string(pose_id) << " ";
        for (auto const& keypoint : pose.keypoints) {
            rawPose << keypoint.x << "," << keypoint.y << " ";
            }
        rawPose << pose.score;
        publisher->send_message(rawPose.str().c_str());
        //std::cout << rawPose.str() << std::endl;
        pose_id++;
    }
}
}  // namespace human_pose_estimation
