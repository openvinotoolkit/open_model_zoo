// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <opencv2/core/core.hpp>

#include "human_pose.hpp"

namespace human_pose_estimation {
struct Peak {
    Peak(const int id = -1,
         const cv::Point2f& pos = cv::Point2f(),
         const float score = 0.0f);

    int id;
    cv::Point2f pos;
    float score;
};

struct HumanPoseByPeaksIndices {
    explicit HumanPoseByPeaksIndices(const int keypointsNumber);

    std::vector<int> peaksIndices;
    int nJoints;
    float score;
};

struct TwoJointsConnection {
    TwoJointsConnection(const int firstJointIdx,
                        const int secondJointIdx,
                        const float score);

    int firstJointIdx;
    int secondJointIdx;
    float score;
};

void findPeaks(const std::vector<cv::Mat>& heatMaps,
               const float minPeaksDistance,
               std::vector<std::vector<Peak> >& allPeaks,
               int heatMapId);

std::vector<HumanPose> groupPeaksToPoses(
        const std::vector<std::vector<Peak> >& allPeaks,
        const std::vector<cv::Mat>& pafs,
        const size_t keypointsNumber,
        const float midPointsScoreThreshold,
        const float foundMidPointsRatioThreshold,
        const int minJointsNumber,
        const float minSubsetScore);
} // namespace human_pose_estimation

