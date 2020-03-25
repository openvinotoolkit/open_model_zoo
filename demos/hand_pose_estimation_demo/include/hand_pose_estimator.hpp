// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include <inference_engine.hpp>
#include <opencv2/core/core.hpp>

#include "hand_pose.hpp"

namespace hand_pose_estimation {
class handPoseEstimator {
public:
    static const size_t keypointsNumber;

    handPoseEstimator(const std::string& modelPath,
                       const std::string& targetDeviceName,
                       bool enablePerformanceReport = false);
    std::vector<handPose> estimate(const cv::Mat& image);
    ~handPoseEstimator();

private:
    void preprocess(const cv::Mat& image, float* buffer) const;
    std::vector<handPose> postprocess(
            const float* heatMapsData, const int heatMapOffset, const int nHeatMaps,
            const float* pafsData, const int pafOffset, const int nPafs,
            const int featureMapWidth, const int featureMapHeight,
            const cv::Size& imageSize) const;
    std::vector<handPose> extractPoses(const std::vector<cv::Mat>& heatMaps,
                                        const std::vector<cv::Mat>& pafs) const;
    void resizeFeatureMaps(std::vector<cv::Mat>& featureMaps) const;
    void correctCoordinates(std::vector<handPose>& poses,
                            const cv::Size& featureMapsSize,
                            const cv::Size& imageSize) const;
    bool inputWidthIsChanged(const cv::Size& imageSize);

    int minJointsNumber;
    int stride;
    cv::Vec4i pad;
    cv::Vec3f meanPixel;
    float minPeaksDistance;
    float midPointsScoreThreshold;
    float foundMidPointsRatioThreshold;
    float minSubsetScore;
    cv::Size inputLayerSize;
    int upsampleRatio;
    InferenceEngine::Core ie;
    std::string targetDeviceName;
    InferenceEngine::CNNNetwork network;
    InferenceEngine::ExecutableNetwork executableNetwork;
    InferenceEngine::InferRequest request;
    InferenceEngine::CNNNetReader netReader;
    std::string pafsBlobName;
    std::string heatmapsBlobName;
    bool enablePerformanceReport;
    std::string modelPath;
};
}  // namespace hand_pose_estimation
