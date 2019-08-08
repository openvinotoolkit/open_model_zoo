// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

#include "core.hpp"
#include "cnn.hpp"


struct DetectorConfig : public CnnConfig {
    explicit DetectorConfig(const std::string& path_to_model,
                            const std::string& path_to_weights)
        : CnnConfig(path_to_model, path_to_weights) {}

    float confidence_threshold{0.5f};
    float increase_scale_x{1.f};
    float increase_scale_y{1.f};
    bool is_async = false;
};

class ObjectDetector {
private:
    InferenceEngine::InferRequest::Ptr request;
    DetectorConfig config_;
    InferenceEngine::Core ie_;
    std::string deviceName_;

    InferenceEngine::ExecutableNetwork net_;
    std::string input_name_;
    std::string im_info_name_;
    std::string output_name_;
    int max_detections_count_;
    int object_size_;
    int enqueued_frames_ = 0;
    float width_ = 0;
    float height_ = 0;
    bool results_fetched_ = false;
    int frame_idx_ = -1;

    TrackedObjects results_;

    void enqueue(const cv::Mat &frame);
    void submitRequest();
    void wait();
    void fetchResults();

public:
    ObjectDetector(const DetectorConfig& config,
                   const InferenceEngine::Core& ie,
                   const std::string & deviceName);

    void submitFrame(const cv::Mat &frame, int frame_idx);
    void waitAndFetchResults();

    const TrackedObjects& getResults() const;

    void PrintPerformanceCounts(std::string fullDeviceName);
};
