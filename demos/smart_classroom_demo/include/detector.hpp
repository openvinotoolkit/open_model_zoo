// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

#include "cnn.hpp"

namespace detection {

struct DetectedObject {
    cv::Rect rect;
    float confidence;

    explicit DetectedObject(const cv::Rect& rect = cv::Rect(), float confidence = -1.0f)
        : rect(rect), confidence(confidence) {}
};

using DetectedObjects = std::vector<DetectedObject>;

struct DetectorConfig : public CnnConfig {
    explicit DetectorConfig(const std::string& path_to_model)
        : CnnConfig(path_to_model) {}

    float confidence_threshold{0.6f};
    float increase_scale_x{1.15f};
    float increase_scale_y{1.15f};
    bool is_async = true;
    int input_h = 600;
    int input_w = 600;
};

class FaceDetection : public AsyncDetection<DetectedObject>, public BaseCnnDetection {
private:
    DetectorConfig config_;
    InferenceEngine::ExecutableNetwork net_;
    std::string input_name_;
    std::string output_name_;
    int max_detections_count_ = 0;
    int object_size_ = 0;
    int enqueued_frames_ = 0;
    float width_ = 0;
    float height_ = 0;

public:
    explicit FaceDetection(const DetectorConfig& config);

    void submitRequest() override;
    void enqueue(const cv::Mat &frame) override;
    void wait() override { BaseCnnDetection::wait(); }
    void printPerformanceCounts(const std::string &fullDeviceName) override {
        BaseCnnDetection::printPerformanceCounts(fullDeviceName);
    }

    DetectedObjects fetchResults() override;
};

}  // namespace detection
