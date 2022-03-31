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

class GazeEstimator : public BaseEstimator {
public:
    GazeEstimator(ov::Core& core, const std::string& modelPath, const std::string& deviceName, bool doRollAlign = true);
    void estimate(const cv::Mat& image, FaceInferenceResults& outputResults) override;
    ~GazeEstimator() override;

    const std::string modelType = "Gaze Estimation";

private:
    IEWrapper ieWrapper;
    std::string outputTensorName;
    bool rollAlign;

    void rotateImageAroundCenter(const cv::Mat& srcImage, cv::Mat& dstImage, float angle) const;
};
}  // namespace gaze_estimation
