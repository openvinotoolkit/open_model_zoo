// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <opencv2/core/core.hpp>

namespace gaze_estimation {
struct FaceInferenceResults {
    float faceDetectionConfidence;
    cv::Rect faceBoundingBox;

    std::vector<cv::Point2i> faceLandmarks;

    cv::Point3f headPoseAngles;

    cv::Rect leftEyeBoundingBox;
    cv::Rect rightEyeBoundingBox;
    cv::Point2f leftEyeMidpoint;
    cv::Point2f rightEyeMidpoint;

    cv::Point3f gazeVector;

    friend std::ostream& operator<<(std::ostream& os, const FaceInferenceResults& faceInferenceResults);
};
}  // namespace gaze_estimation
