// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include <inference_engine.hpp>
#include <opencv2/core/core.hpp>

#include "human_pose.hpp"

namespace human_pose_estimation {
class HumanPoseEstimator {
public:
    static const size_t keypointsNumber = 18;

    HumanPoseEstimator(const std::string& modelPath,
                       const std::string& targetDeviceName,
                       bool enablePerformanceReport = false);
    std::vector<HumanPose> postprocessCurr();
    void reshape(const cv::Mat& image);
    void frameToBlobCurr(const cv::Mat& image);
    void frameToBlobNext(const cv::Mat& image);
    void startCurr();
    void startNext();
    bool readyCurr();
    void swapRequest();
    ~HumanPoseEstimator();

private:
    void preprocess(const cv::Mat& image, uint8_t* buffer) const;
    std::vector<HumanPose> postprocess(
            const float* heatMapsData, const int heatMapOffset, const int nHeatMaps,
            const float* pafsData, const int pafOffset, const int nPafs,
            const int featureMapWidth, const int featureMapHeight,
            const cv::Size& imageSize) const;
    std::vector<HumanPose> extractPoses(const std::vector<cv::Mat>& heatMaps,
                                        const std::vector<cv::Mat>& pafs) const;
    void resizeFeatureMaps(std::vector<cv::Mat>& featureMaps) const;
    void correctCoordinates(std::vector<HumanPose>& poses,
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
    cv::Size imageSize;
    int upsampleRatio;
    InferenceEngine::Core ie;
    std::string targetDeviceName;
    InferenceEngine::CNNNetwork network;
    InferenceEngine::ExecutableNetwork executableNetwork;
    InferenceEngine::InferRequest::Ptr requestNext;
    InferenceEngine::InferRequest::Ptr requestCurr;
    std::string pafsBlobName;
    std::string heatmapsBlobName;
    bool enablePerformanceReport;
    std::string modelPath;
};
}  // namespace human_pose_estimation
