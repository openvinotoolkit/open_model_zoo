/*
// Copyright (C) 2021 Intel Corporation
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

#include <algorithm>
#include <utility>
#include <vector>

#include <utils/common.hpp>
#include "utils/kuhn_munkres.hpp"

#include "models/associative_embedding_decoder.h"


void findPeaks(const std::vector<cv::Mat>& nmsHeatMaps,
               const std::vector<cv::Mat>& aembdsMaps,
               std::vector<std::vector<Peak>>& allPeaks,
               int jointId, const int maxNumPeople,
               float detectionThreshold) {

    const cv::Mat& nmsHeatMap = nmsHeatMaps[jointId];
    const float* heatMapData = nmsHeatMap.ptr<float>();
    cv::Size outputSize = nmsHeatMap.size();

    std::vector<int> indices(outputSize.area());
    std::iota(std::begin(indices), std::end(indices), 0);
    std::partial_sort(std::begin(indices), std::begin(indices) + maxNumPeople, std::end(indices),
                      [heatMapData](int l, int r) { return heatMapData[l] > heatMapData[r]; });

    for (int personId = 0; personId < maxNumPeople; personId++) {
        int index = indices[personId];
        int x = index / outputSize.width;
        int y = index % outputSize.width;
        float tag = aembdsMaps[jointId].at<float>(x, y);
        float score = heatMapData[index];
        allPeaks[jointId].reserve(maxNumPeople);
        if (score > detectionThreshold) {
            allPeaks[jointId].emplace_back(Peak{cv::Point2f(static_cast<float>(x), static_cast<float>(y)),
                                           score, tag});
        }
    }
}

std::vector<Pose> matchByTag(std::vector<std::vector<Peak>>& allPeaks,
                             int maxNumPeople, int numJoints,
                             float tagThreshold,
                             bool useDetectionVal, bool ignoreTooMuch) {
    size_t jointOrder[] { 0, 1, 2, 3, 4, 5, 6, 11, 12, 7, 8, 9, 10, 13, 14, 15, 16 };
    std::vector<Pose> allPoses;
    for (size_t jointId : jointOrder) {
        std::vector<Peak>& jointPeaks = allPeaks[jointId];
        std::vector<float> tags;
        for (auto& peak: jointPeaks) {
            tags.push_back(peak.tag);
        }
        if (allPoses.empty()) {
            for (size_t personId = 0; personId < jointPeaks.size(); personId++) {
                Peak peak = jointPeaks[personId];
                Pose pose = Pose(numJoints);
                pose.add(jointId, peak);
                allPoses.push_back(pose);
            }
            continue;
        }
        if (jointPeaks.empty() || (ignoreTooMuch && allPoses.size() == maxNumPeople)) {
            continue;
        }
        std::vector<float> posesTags;
        for (auto& pose : allPoses) {
            posesTags.push_back(pose.getPoseTag());
        }
        // Compute dissimilarity matrix
        int numAdded = tags.size();
        int numGrouped = posesTags.size();
        cv::Mat dissimilarity(numAdded, numGrouped, CV_32F);
        cv::Mat dissimilarityCopy(numAdded, numGrouped, CV_32F);
        for (int i = 0; i < dissimilarity.rows; i++) {
            for (int j = 0; j < dissimilarity.cols; j++) {
                float diffNormed = static_cast<float>(cv::norm(tags[i] - posesTags[j]));
                dissimilarityCopy.at<float>(i, j) = diffNormed;
                if (useDetectionVal) {
                    diffNormed = std::round(diffNormed) * 100 - jointPeaks[i].score;
                }
                dissimilarity.at<float>(i, j) = diffNormed;
            }
        }
        if (numAdded > numGrouped) {
            cv::copyMakeBorder(dissimilarity, dissimilarity, 0, 0, 0, numAdded - numGrouped,
                               cv::BORDER_CONSTANT, 10000000000);
        }
        // Get pairs
        auto res = KuhnMunkres().Solve(dissimilarity);
        for (int row = 0; row < res.size(); row++) {
            int col = res[row];
            if (row < numAdded && col < numGrouped && dissimilarityCopy.at<float>(row, col) < tagThreshold) {
                allPoses[col].add(jointId, jointPeaks[row]);
            }
            else {
                Pose pose = Pose(numJoints);
                pose.add(jointId, jointPeaks[row]);
                allPoses.push_back(pose);
            }
        }
    }
    return allPoses;
}

void adjustAndRefine(std::vector<Pose>& allPoses,
                     const std::vector<cv::Mat>& heatMaps,
                     const std::vector<cv::Mat>& aembdsMaps,
                     int poseId, const float delta,
                     bool doAdjust, bool doRefine) {

    cv::Size outputSize = heatMaps[0].size();
    Pose& pose = allPoses[poseId];
    float poseTag = pose.getPoseTag();

    for (size_t jointId = 0; jointId < pose.size(); jointId++) {
        Peak& peak = pose.getPeak(jointId);
        const cv::Mat& heatMap = heatMaps[jointId];
        const cv::Mat& aembds = aembdsMaps[jointId];

        if (doAdjust) {
            int px = static_cast<int>(peak.keypoint.x);
            int py = static_cast<int>(peak.keypoint.y);
            if ((1 < px) && (px < outputSize.width - 1) && (1 < py) && (py < outputSize.height - 1)) {
                auto diffX = heatMap.at<float>(py, px + 1) - heatMap.at<float>(py, px - 1);
                auto diffY = heatMap.at<float>(py + 1, px) - heatMap.at<float>(py - 1, px);
                peak.keypoint.x += diffX > 0 ? 0.25f : -0.25f;
                peak.keypoint.y += diffY > 0 ? 0.25f : -0.25f;
            }
            if (delta) {
                peak.keypoint.x += delta;
                peak.keypoint.y += delta;
            }
        }
        if (doRefine && peak.score > 0) {
            float minValue = std::numeric_limits<float>::max();
            int x, y;
            // Get position with the closest tag value to the pose tag
            for (int i = 0; i < outputSize.height; i++) {
                for (int j = 0; j < outputSize.width; j++) {
                    float diff = std::abs(aembds.at<float>(i, j) - poseTag) + 0.5f;
                    diff -= heatMap.at<float>(i, j);
                    if (diff < minValue) {
                        minValue = diff;
                        x = i;
                        y = j;
                    }
                }
            }
            float val = heatMap.at<float>(x, y);
            if (val > 0) {
                peak.keypoint.x = static_cast<float>(x);
                peak.keypoint.y = static_cast<float>(y);
                peak.score = val;
                if ((1 < x) && (x < outputSize.width - 1) && (1 < y) && (y < outputSize.height - 1)) {
                    auto diffX = heatMap.at<float>(y, x + 1) - heatMap.at<float>(y, x - 1);
                    auto diffY = heatMap.at<float>(y + 1, x) - heatMap.at<float>(y - 1, x);
                    peak.keypoint.x += diffX > 0 ? 0.25f : -0.25f;
                    peak.keypoint.y += diffY > 0 ? 0.25f : -0.25f;
                }
            }
        }
    }
}
