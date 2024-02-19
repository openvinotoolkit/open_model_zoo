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
class HeadPoseEstimator: public BaseEstimator {
public:
    HeadPoseEstimator(ov::Core& core,
                      const std::string& modelPath,
                      const std::string& deviceName);
    void estimate(const cv::Mat& image,
                  FaceInferenceResults& outputResults) override;
    ~HeadPoseEstimator() override;

    const std::string modelType = "Head Pose Estimation";

private:
    IEWrapper ieWrapper;
    std::string inputTensorName;
};
}  // namespace gaze_estimation
