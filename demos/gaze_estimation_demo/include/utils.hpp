// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>
#include <string>

#include <utility>
#include <map>
#include <vector>

#include <inference_engine.hpp>

#include <ie_iextension.h>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

namespace gaze_estimation {
void initializeIEObject(InferenceEngine::Core& ie,
                        const std::vector<std::pair<std::string, std::string>>& cmdOptions);

void gazeVectorToGazeAngles(const cv::Point3f& gazeVector, cv::Point2f& gazeAngles);

void putTimingInfoOnFrame(cv::Mat& image, double overallTime, double inferenceTime);
}  // namespace gaze_estimation
