// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>
#include <string>

#include "face_inference_results.hpp"

namespace gaze_estimation {
class ResultsMarker {
public:
    ResultsMarker(bool showFaceBoundingBox,
                  bool showHeadPoseAxes,
                  bool showLandmarks,
                  bool showGaze,
                  bool showEyeState);
    void mark(cv::Mat& image, const FaceInferenceResults& faceInferenceResults) const;
    void toggle(int key);

private:
    bool showFaceBoundingBox;
    bool showHeadPoseAxes;
    bool showLandmarks;
    bool showGaze;
    bool showEyeState;
};
}  // namespace gaze_estimation
