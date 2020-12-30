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

#include <samples/common.hpp>
#include "models/associative_embedding_decoder.h"
#include "models/kuhn_munkres.hpp"


void findPeaks(const std::vector<cv::Mat>& nmsHeatMaps,
               const std::vector<cv::Mat>& aembdsMaps,
               std::vector<std::vector<Peak>>& allPeaks,
               int jointId, const int maxNumPeople) {
    
    const cv::Mat& nmsHeatMap = nmsHeatMaps[jointId];
    const float* heatMapData = nmsHeatMap.ptr<float>();
    cv::Size outputSize = nmsHeatMap.size();
    size_t heatMapStep = nmsHeatMap.step1();

    std::vector<float> peaks(outputSize.area());
    for (int i = 0; i < outputSize.height; i++) {
        for (int j = 0; j < outputSize.width; j++) {
            size_t index = i * heatMapStep + j;
            peaks[index] = heatMapData[index];
        }
    }
    std::vector<int> indices(outputSize.area());
    std::iota(std::begin(indices), std::end(indices), 0);
    std::partial_sort(std::begin(indices), std::begin(indices) + maxNumPeople, std::end(indices),
                      [&peaks](int l, int r) { return peaks[l] > peaks[r]; });
    for (int personId = 0; personId < maxNumPeople; personId++) {
        int index = indices[personId];
        int x = index / outputSize.width;
        int y = index % outputSize.width;
        float tag = aembdsMaps[jointId].at<float>(x, y);
        allPeaks[jointId].push_back(Peak{cv::Point2f(static_cast<float>(x), static_cast<float>(y)),
                                         peaks[index], tag});
    }
}

std::vector<Pose> matchByTag(std::vector<std::vector<Peak>>& allPeaks,
                             int maxNumPeople,
                             float detectionThreshold,
                             float tagThreshold,
                             int numJoints,
                             bool useDetectionVal,
                             bool ignoreTooMuch) {
    std::vector<size_t> jointOrder { 0, 1, 2, 3, 4, 5, 6, 11, 12, 7, 8, 9, 10, 13, 14, 15, 16 };
    std::vector<Pose> allPoses;
    for (size_t i = 0; i < numJoints; i++) {
        size_t jointId = jointOrder[i];
        std::vector<Peak>& jointPeaks = allPeaks[jointId];
        std::vector<float> tags;
        auto it = std::begin(jointPeaks);
        // Filtering peaks with low scores
        while (it != std::end(jointPeaks)) {
            Peak peak = *it;
            if (peak.score <= detectionThreshold) {
                it = jointPeaks.erase(it);
            }
            else {
                tags.push_back(peak.tag);
                ++it;
            }
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
                float diff = tags[i] - posesTags[j];
                float diffNormed = static_cast<float>(cv::norm(diff));
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
    Pose pose = allPoses[poseId];
    float poseTag = pose.getPoseTag();

    for (size_t jointId = 0; jointId < pose.size(); jointId++) {
        Peak peak = pose.getPeak(jointId);
        const cv::Mat& peaks = heatMaps[jointId];
        const cv::Mat& aembds = aembdsMaps[jointId];

        if (doAdjust) {
            int px = static_cast<int>(peak.pos.x);
            int py = static_cast<int>(peak.pos.y);
            if ((1 < px) && (px < outputSize.width - 1) && (1 < py) && (py < outputSize.height - 1)) {
                auto diffX = peaks.at<float>(py, px + 1) - peaks.at<float>(py, px - 1);
                auto diffY = peaks.at<float>(py + 1, px) - peaks.at<float>(py - 1, px);
                peak.pos.x += diffX > 0 ? 0.25f : -0.25f; // static_cast<float>(
                peak.pos.y += diffY > 0 ? 0.25f : -0.25f;
            }
            if (delta) {
                peak.pos.x += delta;
                peak.pos.y += delta;
            }
        }
        if (doRefine && peak.score > 0) {
            float minValue = std::numeric_limits<float>::max();
            int x, y;
            // Get position with the closest tag value to the pose tag.
            for (size_t row = 0; row < aembds.rows; row++) {
                for (size_t col = 0; col < aembds.cols; col++) {
                    float diff = static_cast<float>(std::abs(aembds.at<float>(row, col) - poseTag) + 0.5);
                    diff -= peaks.at<float>(row, col);
                    if (diff < minValue) {
                        minValue = diff;
                        x = row;
                        y = col;
                    }
                }
            }
            float val = peaks.at<float>(x, y);
            if (val > 0) {
                peak.pos.x = static_cast<float>(x);
                peak.pos.y = static_cast<float>(y);
                peak.score = val;
                if ((1 < x) && (x < outputSize.width - 1) && (1 < y) && (y < outputSize.height - 1)) {
                    auto diffX = peaks.at<float>(y, x + 1) - peaks.at<float>(y, x - 1);
                    auto diffY = peaks.at<float>(y + 1, x) - peaks.at<float>(y - 1, x);
                    peak.pos.x += diffX > 0 ? 0.25f : -0.25f;
                    peak.pos.y += diffY > 0 ? 0.25f : -0.25f;
                }
            }
        }
    }
}
