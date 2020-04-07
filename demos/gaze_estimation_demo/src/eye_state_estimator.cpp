// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "eye_state_estimator.hpp"

namespace gaze_estimation {

EyeStateEstimator::EyeStateEstimator(InferenceEngine::Core& ie,
                                     const std::string& modelPath,
                                     const std::string& deviceName):
                                     ieWrapper(ie, modelPath, deviceName) {
    inputBlobName = ieWrapper.expectSingleInput();
    ieWrapper.expectImageInput(inputBlobName);
    outputBlobName = ieWrapper.expectSingleOutput();
    const auto& outputInfo = ieWrapper.getOutputBlobDimsInfo();
}

void EyeStateEstimator::estimate(const cv::Mat& image, FaceInferenceResults& outputResults) {

    auto leftEyeBoundingBox = outputResults.leftEyeBoundingBox;
    auto leftEyeCrop(cv::Mat(image, leftEyeBoundingBox));
    auto rightEyeBoundingBox = outputResults.rightEyeBoundingBox;
    auto rightEyeCrop(cv::Mat(image, rightEyeBoundingBox));

    std::vector<float> outputValue;
    ieWrapper.setInputBlob(inputBlobName, leftEyeCrop);
    ieWrapper.infer();
    ieWrapper.getOutputBlob(outputBlobName, outputValue);
    outputResults.leftEyeState = outputValue[0] < outputValue[1];

    outputValue.clear();
    ieWrapper.setInputBlob(inputBlobName, rightEyeCrop);
    ieWrapper.infer();
    ieWrapper.getOutputBlob(outputBlobName, outputValue);
    outputResults.rightEyeState = outputValue[0] < outputValue[1];  
}

void EyeStateEstimator::printPerformanceCounts() const {
    ieWrapper.printPerlayerPerformance();
}

EyeStateEstimator::~EyeStateEstimator() {
}
}  // namespace gaze_estimation
