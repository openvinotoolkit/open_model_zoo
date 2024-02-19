// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include "face_inference_results.hpp"

namespace gaze_estimation {

slog::LogStream& operator<<(slog::LogStream& os, const FaceInferenceResults& faceInferenceResults) {
    os << "--Inference results--" << slog::endl;

    os << "Face detection confidence: " << faceInferenceResults.faceDetectionConfidence << slog::endl;
    os << "Face bounding box: " << faceInferenceResults.faceBoundingBox << slog::endl;

    os << "Facial landmarks: " << slog::endl;
    int lmCounter = 0;
    for (auto const& lm : faceInferenceResults.faceLandmarks) {
        os << "\tlandmark #" << lmCounter << ": " << lm << slog::endl;
        ++lmCounter;
    }

    os << "Head pose angles (yaw, pitch, roll): " << faceInferenceResults.headPoseAngles << slog::endl;
    os << "Gaze vector (x, y, z): " << faceInferenceResults.gazeVector << slog::endl;

    os << "--End of inference results--" << slog::endl;

    return os;
}

std::vector<cv::Point2f> FaceInferenceResults::getEyeLandmarks() {
    std::vector<cv::Point2f> result(4);
    if (faceLandmarks.size() == 35) {
        result[0] = faceLandmarks[0];
        result[1] = faceLandmarks[1];
        result[2] = faceLandmarks[2];
        result[3] = faceLandmarks[3];
    }
    else if (faceLandmarks.size() == 98) {
        result[0] = faceLandmarks[60];
        result[1] = faceLandmarks[64];
        result[2] = faceLandmarks[68];
        result[3] = faceLandmarks[72];
    }
    else {
        throw std::runtime_error("the network must output 35 or 98 points");
    }
    return result;
}
}  // namespace gaze_estimation
