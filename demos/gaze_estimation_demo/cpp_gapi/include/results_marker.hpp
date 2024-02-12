// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace cv {
class Mat;
}  // namespace cv

namespace gaze_estimation {
struct FaceInferenceResults;

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
