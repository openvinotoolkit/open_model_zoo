// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>
#include <string>

#include "face_inference_results.hpp"
#include "base_estimator.hpp"

#include "ie_wrapper.hpp"

namespace gaze_estimation {
class GazeEstimator: public BaseEstimator {
public:
    GazeEstimator(InferenceEngine::Core& ie,
                  const std::string& modelPath,
                  const std::string& deviceName,
                  bool doRollAlign = true);
    void estimate(const cv::Mat& image,
                  FaceInferenceResults& outputResults) override;
    void printPerformanceCounts() const override;
    ~GazeEstimator() override;

private:
    IEWrapper ieWrapper;
    std::string outputBlobName;
    bool rollAlign;

    void rotateImageAroundCenter(const cv::Mat& srcImage, cv::Mat& dstImage, float angle) const;
};
}  // namespace gaze_estimation
