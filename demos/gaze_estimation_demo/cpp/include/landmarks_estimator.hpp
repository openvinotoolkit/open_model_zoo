// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>
#include <string>

#include "face_inference_results.hpp"
#include "base_estimator.hpp"

#include "ie_wrapper.hpp"

namespace gaze_estimation {
class LandmarksEstimator: public BaseEstimator {
public:
    LandmarksEstimator(ov::Core& core,
                       const std::string& modelPath,
                       const std::string& deviceName);
    void estimate(const cv::Mat& image,
                  FaceInferenceResults& outputResults) override;
    ~LandmarksEstimator() override;

    const std::string modelType = "Facial Landmarks Estimation";

private:
    IEWrapper ieWrapper;
    std::string inputTensorName, outputTensorName;
    size_t numberLandmarks;
    std::vector<cv::Point2i> simplePostprocess(cv::Rect faceBoundingBox, cv::Mat faceCrop);
    std::vector<cv::Point2i> heatMapPostprocess(cv::Rect faceBoundingBox, cv::Mat faceCrop);
    std::vector<cv::Mat> split(std::vector<float>& data, const ov::Shape& shape);
    std::vector<cv::Point2f> getMaxPreds(std::vector<cv::Mat> heatMaps);
    int sign(float number);
    cv::Mat affineTransform(cv::Point2f center, cv::Point2f scale,
        float rot, size_t dst_w, size_t dst_h, cv::Point2f shift, bool inv);
    cv::Point2f rotatePoint(cv::Point2f pt, float angle_rad);
    cv::Point2f get3rdPoint(cv::Point2f a, cv::Point2f b);
};
}  // namespace gaze_estimation
