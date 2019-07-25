// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include "face_inference_results.hpp"

namespace gaze_estimation {

std::ostream& operator<<(std::ostream& os, const FaceInferenceResults& faceInferenceResults) {
    os << "--Inference results--" << std::endl;

    os << "Face detection confidence: " << faceInferenceResults.faceDetectionConfidence << std::endl;
    os << "Face bounding box: " << faceInferenceResults.faceBoundingBox << std::endl;

    os << "Facial landmarks: " << std::endl;
    int lmCounter = 0;
    for (auto const& lm : faceInferenceResults.faceLandmarks) {
        os << "\t landmark #" << lmCounter << ": " << lm << std::endl;
        ++lmCounter;
    }

    os << "Head pose angles (yaw, pitch, roll): " << faceInferenceResults.headPoseAngles << std::endl;
    os << "Gaze vector (x, y, z): " << faceInferenceResults.gazeVector << std::endl;

    os << "--End of inference results--" << std::endl;

    return os;
}

}  // namespace gaze_estimation
