// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iosfwd>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>

#include "ie_wrapper.hpp"

namespace gaze_estimation {
struct FaceInferenceResults;

class FaceDetector {
public:
    FaceDetector(ov::Core& core,
                 const std::string& modelPath,
                 const std::string& deviceName,
                 double detectionConfidenceThreshold,
                 bool enableReshape);
    std::vector<FaceInferenceResults> detect(const cv::Mat& image);
    ~FaceDetector();

    const std::string modelType = "Face Detection";

private:
    IEWrapper ieWrapper;
    std::string inputTensorName;
    ov::Shape inputTensorDims;
    std::string outputTensorName;
    std::size_t numTotalDetections;

    double detectionThreshold;
    bool enableReshape;

    void adjustBoundingBox(cv::Rect& boundingBox) const;
};
}  // namespace gaze_estimation
