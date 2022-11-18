/*
// Copyright (C) 2020-2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#pragma once
#include <stddef.h>

#include <vector>

#include <opencv2/core.hpp>

struct HumanPose;

struct Peak {
    Peak(const int id = -1, const cv::Point2f& pos = cv::Point2f(), const float score = 0.0f);

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
    TwoJointsConnection(const int firstJointIdx, const int secondJointIdx, const float score);

    int firstJointIdx;
    int secondJointIdx;
    float score;
};

void findPeaks(const std::vector<cv::Mat>& heatMaps,
               const float minPeaksDistance,
               std::vector<std::vector<Peak>>& allPeaks,
               int heatMapId,
               float confidenceThreshold);

std::vector<HumanPose> groupPeaksToPoses(const std::vector<std::vector<Peak>>& allPeaks,
                                         const std::vector<cv::Mat>& pafs,
                                         const size_t keypointsNumber,
                                         const float midPointsScoreThreshold,
                                         const float foundMidPointsRatioThreshold,
                                         const int minJointsNumber,
                                         const float minSubsetScore);
