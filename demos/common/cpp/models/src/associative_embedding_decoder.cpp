/*
// Copyright (C) 2021-2022 Intel Corporation
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

#include "models/associative_embedding_decoder.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <vector>

#include <utils/kuhn_munkres.hpp>

void findPeaks(const std::vector<cv::Mat>& nmsHeatMaps,
               const std::vector<cv::Mat>& aembdsMaps,
               std::vector<std::vector<Peak>>& allPeaks,
               size_t jointId,
               size_t maxNumPeople,
               float detectionThreshold) {
    const cv::Mat& nmsHeatMap = nmsHeatMaps[jointId];
    const float* heatMapData = nmsHeatMap.ptr<float>();
    cv::Size outputSize = nmsHeatMap.size();

    std::vector<int> indices(outputSize.area());
    std::iota(std::begin(indices), std::end(indices), 0);
    std::partial_sort(std::begin(indices),
                      std::begin(indices) + maxNumPeople,
                      std::end(indices),
                      [heatMapData](int l, int r) {
                          return heatMapData[l] > heatMapData[r];
                      });

    for (size_t personId = 0; personId < maxNumPeople; personId++) {
        int index = indices[personId];
        int x = index / outputSize.width;
        int y = index % outputSize.width;
        float tag = aembdsMaps[jointId].at<float>(x, y);
        float score = heatMapData[index];
        allPeaks[jointId].reserve(maxNumPeople);
        if (score > detectionThreshold) {
            allPeaks[jointId].emplace_back(Peak{cv::Point2f(static_cast<float>(x), static_cast<float>(y)), score, tag});
        }
    }
}

std::vector<Pose> matchByTag(std::vector<std::vector<Peak>>& allPeaks,
                             size_t maxNumPeople,
                             size_t numJoints,
                             float tagThreshold) {
    size_t jointOrder[]{0, 1, 2, 3, 4, 5, 6, 11, 12, 7, 8, 9, 10, 13, 14, 15, 16};
    std::vector<Pose> allPoses;
    for (size_t jointId : jointOrder) {
        std::vector<Peak>& jointPeaks = allPeaks[jointId];
        std::vector<float> tags;
        for (auto& peak : jointPeaks) {
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
        if (jointPeaks.empty() || (allPoses.size() == maxNumPeople)) {
            continue;
        }
        std::vector<float> posesTags;
        std::vector<cv::Point2f> posesCenters;
        for (auto& pose : allPoses) {
            posesTags.push_back(pose.getPoseTag());
            posesCenters.push_back(pose.getPoseCenter());
        }
        size_t numAdded = tags.size();
        size_t numGrouped = posesTags.size();
        cv::Mat tagsDiff(numAdded, numGrouped, CV_32F);
        cv::Mat matchingCost(numAdded, numGrouped, CV_32F);
        std::vector<float> dists(numAdded);
        for (size_t j = 0; j < numGrouped; j++) {
            float minDist = std::numeric_limits<float>::max();
            // Compute euclidean distance (in spatial space) between the pose center and all joints.
            const cv::Point2f center = posesCenters.at(j);
            for (size_t i = 0; i < numAdded; i++) {
                cv::Point2f v = jointPeaks.at(i).keypoint - center;
                float dist = std::sqrt(v.x * v.x + v.y * v.y);
                dists[i] = dist;
                minDist = std::min(dist, minDist);
            }
            // Compute semantic distance (in embedding space) between the pose tag and all joints
            // and corresponding matching costs.
            auto poseTag = posesTags[j];
            for (size_t i = 0; i < numAdded; i++) {
                float diff = static_cast<float>(cv::norm(tags[i] - poseTag));
                tagsDiff.at<float>(i, j) = diff;
                if (diff < tagThreshold) {
                    diff *= dists[i] / (minDist + 1e-10f);
                }
                matchingCost.at<float>(i, j) = std::round(diff) * 100 - jointPeaks[i].score;
            }
        }

        if (numAdded > numGrouped) {
            cv::copyMakeBorder(matchingCost,
                               matchingCost,
                               0,
                               0,
                               0,
                               numAdded - numGrouped,
                               cv::BORDER_CONSTANT,
                               10000000);
        }
        // Get pairs
        auto res = KuhnMunkres().Solve(matchingCost);
        for (size_t row = 0; row < res.size(); row++) {
            size_t col = res[row];
            if (row < numAdded && col < numGrouped && tagsDiff.at<float>(row, col) < tagThreshold) {
                allPoses[col].add(jointId, jointPeaks[row]);
            } else {
                Pose pose = Pose(numJoints);
                pose.add(jointId, jointPeaks[row]);
                allPoses.push_back(pose);
            }
        }
    }
    return allPoses;
}

namespace {
cv::Point2f adjustLocation(const int x, const int y, const cv::Mat& heatMap) {
    cv::Point2f delta(0.f, 0.f);
    int width = heatMap.cols;
    int height = heatMap.rows;
    if ((1 < x) && (x < width - 1) && (1 < y) && (y < height - 1)) {
        auto diffX = heatMap.at<float>(y, x + 1) - heatMap.at<float>(y, x - 1);
        auto diffY = heatMap.at<float>(y + 1, x) - heatMap.at<float>(y - 1, x);
        delta.x = diffX > 0 ? 0.25f : -0.25f;
        delta.y = diffY > 0 ? 0.25f : -0.25f;
    }
    return delta;
}
}  // namespace

void adjustAndRefine(std::vector<Pose>& allPoses,
                     const std::vector<cv::Mat>& heatMaps,
                     const std::vector<cv::Mat>& aembdsMaps,
                     int poseId,
                     const float delta) {
    Pose& pose = allPoses[poseId];
    float poseTag = pose.getPoseTag();
    for (size_t jointId = 0; jointId < pose.size(); jointId++) {
        Peak& peak = pose.getPeak(jointId);
        const cv::Mat& heatMap = heatMaps[jointId];
        const cv::Mat& aembds = aembdsMaps[jointId];

        if (peak.score > 0) {
            // Adjust
            int x = static_cast<int>(peak.keypoint.x);
            int y = static_cast<int>(peak.keypoint.y);
            peak.keypoint += adjustLocation(x, y, heatMap);
            if (delta) {
                peak.keypoint.x += delta;
                peak.keypoint.y += delta;
            }
        } else {
            // Refine
            // Get position with the closest tag value to the pose tag
            cv::Mat diff = cv::abs(aembds - poseTag);
            diff.convertTo(diff, CV_32S, 1.0, 0.0);
            diff.convertTo(diff, CV_32F);
            diff -= heatMap;
            double min;
            cv::Point2i minLoc;
            cv::minMaxLoc(diff, &min, 0, &minLoc);
            int x = minLoc.x;
            int y = minLoc.y;
            float val = heatMap.at<float>(y, x);
            if (val > 0) {
                peak.keypoint.x = static_cast<float>(x);
                peak.keypoint.y = static_cast<float>(y);
                peak.keypoint += adjustLocation(x, y, heatMap);
                // Peak score is assigned directly, so it does not affect the pose score.
                peak.score = val;
            }
        }
    }
}
