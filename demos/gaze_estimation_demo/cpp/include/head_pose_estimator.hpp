// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "base_estimator.hpp"
#include "ie_wrapper.hpp"

namespace cv {
class Mat;
}  // namespace cv
namespace ov {
class Core;
}  // namespace ov

namespace gaze_estimation {
struct FaceInferenceResults;

class HeadPoseEstimator : public BaseEstimator {
public:
    HeadPoseEstimator(ov::Core& core, const std::string& modelPath, const std::string& deviceName);
    void estimate(const cv::Mat& image, FaceInferenceResults& outputResults) override;
    ~HeadPoseEstimator() override;

    const std::string modelType = "Head Pose Estimation";

private:
    IEWrapper ieWrapper;
    std::string inputTensorName;
};
}  // namespace gaze_estimation
