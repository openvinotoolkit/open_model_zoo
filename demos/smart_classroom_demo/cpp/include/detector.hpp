// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

#include "openvino/openvino.hpp"

#include "cnn.hpp"

namespace detection {

struct DetectedObject {
    cv::Rect rect;
    float confidence;

    explicit DetectedObject(const cv::Rect& rect = cv::Rect(), float confidence = -1.0f) :
        rect(rect), confidence(confidence) {}
};

using DetectedObjects = std::vector<DetectedObject>;

struct DetectorConfig : public CnnConfig {
    explicit DetectorConfig(const std::string& path_to_model) :
        CnnConfig(path_to_model) {}

    float confidence_threshold{0.6f};
    float increase_scale_x{1.15f};
    float increase_scale_y{1.15f};
    bool is_async = true;
    int input_h = 600;
    int input_w = 600;
};

class FaceDetection : public AsyncDetection<DetectedObject>, public BaseCnnDetection {
private:
    DetectorConfig m_config;
    ov::CompiledModel m_model;
    std::string m_input_name;
    std::string m_output_name;
    int m_max_detections_count = 0;
    int m_object_size = 0;
    int m_enqueued_frames = 0;
    float m_width = 0;
    float m_height = 0;

public:
    explicit FaceDetection(const DetectorConfig& config);

    void submitRequest() override;
    void enqueue(const cv::Mat& frame) override;
    void wait() override { BaseCnnDetection::wait(); }

    DetectedObjects fetchResults() override;
};

}  // namespace detection
