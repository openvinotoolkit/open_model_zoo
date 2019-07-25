// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "landmarks_estimator.hpp"

namespace gaze_estimation {
LandmarksEstimator::LandmarksEstimator(InferenceEngine::Core& ie,
                                       const std::string& modelPath,
                                       const std::string& deviceName):
                    ieWrapper(ie, modelPath, deviceName) {
}

void LandmarksEstimator::estimate(const cv::Mat& image,
                                  FaceInferenceResults& outputResults) {
    auto faceBoundingBox = outputResults.faceBoundingBox;
    auto faceCrop(cv::Mat(image, faceBoundingBox));

    auto inputBlobName = ieWrapper.getIputBlobDimsInfo().begin()->first;

    ieWrapper.setInputBlob(inputBlobName, faceCrop);
    ieWrapper.infer();
    std::vector<float> rawLandmarks;

    ieWrapper.getOutputBlob(rawLandmarks);

    for (unsigned long i = 0; i < rawLandmarks.size() / 2; ++i) {
        int x = static_cast<int>(rawLandmarks[2 * i] * faceCrop.cols + faceBoundingBox.tl().x);
        int y = static_cast<int>(rawLandmarks[2 * i + 1] * faceCrop.rows + faceBoundingBox.tl().y);
        outputResults.faceLandmarks.push_back(cv::Point2i(x, y));
    }
}

void LandmarksEstimator::printPerformanceCounts() const {
    ieWrapper.printPerlayerPerformance();
}

LandmarksEstimator::~LandmarksEstimator() {
}
}  // namespace gaze_estimation
