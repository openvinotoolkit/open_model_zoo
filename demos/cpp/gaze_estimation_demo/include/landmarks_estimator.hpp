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
class LandmarksEstimator: public BaseEstimator {
public:
    LandmarksEstimator(InferenceEngine::Core& ie,
                       const std::string& modelPath,
                       const std::string& deviceName);
    void virtual estimate(const cv::Mat& image,
                          FaceInferenceResults& outputResults);
    void virtual printPerformanceCounts() const;
    virtual ~LandmarksEstimator();

private:
    IEWrapper ieWrapper;
    std::string inputBlobName, outputBlobName;
};
}  // namespace gaze_estimation
