// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace detection {

struct DetectorConfig {
    float confidence_threshold{0.6f};
    float increase_scale_x{1.15f};
    float increase_scale_y{1.15f};
};

class FaceDetection {
private:
    DetectorConfig config_;

public:
    explicit FaceDetection(const DetectorConfig& config) : config_(config) {}
    void truncateRois(const cv::Mat&, const std::vector<cv::Rect>&, std::vector<cv::Rect>&);
};

}  // namespace detection
