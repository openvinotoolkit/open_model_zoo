// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace slog {
class LogStream;
}  // namespace slog

namespace gaze_estimation {
struct FaceInferenceResults {
    float faceDetectionConfidence{};
    cv::Rect faceBoundingBox;

    std::vector<cv::Point2i> faceLandmarks;

    cv::Point3f headPoseAngles;
    cv::Rect leftEyeBoundingBox;
    cv::Rect rightEyeBoundingBox;
    cv::Point2f leftEyeMidpoint;
    cv::Point2f rightEyeMidpoint;
    bool leftEyeState{};
    bool rightEyeState{};

    cv::Point3f gazeVector;

    friend slog::LogStream& operator<<(slog::LogStream& os, const FaceInferenceResults& faceInferenceResults);
};
}  // namespace gaze_estimation
