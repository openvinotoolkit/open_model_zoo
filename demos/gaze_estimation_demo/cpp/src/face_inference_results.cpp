// Copyright (C) 2018 Intel Corporation
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
}  // namespace gaze_estimation
