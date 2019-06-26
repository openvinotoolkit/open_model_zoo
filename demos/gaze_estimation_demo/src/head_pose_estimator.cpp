// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "head_pose_estimator.hpp"

namespace gaze_estimation {
HeadPoseEstimator::HeadPoseEstimator(InferenceEngine::Core& ie,
                                     const std::string& modelPath,
                                     const std::string& deviceName):
                   ieWrapper(ie, modelPath, deviceName) {
}

void HeadPoseEstimator::estimate(const cv::Mat& image,
                                 FaceInferenceResults& outputResults) {
    auto faceBoundingBox = outputResults.faceBoundingBox;
    auto faceCrop(cv::Mat(image, faceBoundingBox));

    auto inputBlobName = ieWrapper.getIputBlobDimsInfo().begin()->first;

    ieWrapper.setInputBlob(inputBlobName, faceCrop);
    ieWrapper.infer();
    std::vector<float> y, p, r;

    ieWrapper.getOutputBlob("angle_y_fc", y);
    ieWrapper.getOutputBlob("angle_p_fc", p);
    ieWrapper.getOutputBlob("angle_r_fc", r);

    outputResults.headPoseAngles.x = y[0];
    outputResults.headPoseAngles.y = p[0];
    outputResults.headPoseAngles.z = r[0];
}

void HeadPoseEstimator::printPerformanceCounts() const {
    ieWrapper.printPerlayerPerformance();
}

HeadPoseEstimator::~HeadPoseEstimator() {
}
}  // namespace gaze_estimation
