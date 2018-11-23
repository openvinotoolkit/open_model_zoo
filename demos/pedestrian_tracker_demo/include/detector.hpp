/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

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
    int input_h = 320;
    int input_w = 544;
};

class ObjectDetector {
private:
    InferenceEngine::InferRequest::Ptr request;
    DetectorConfig config_;
    InferenceEngine::InferencePlugin plugin_;

    InferenceEngine::ExecutableNetwork net_;
    std::string input_name_;
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
    explicit ObjectDetector(const DetectorConfig& config,
                            const InferenceEngine::InferencePlugin& plugin);

    void submitFrame(const cv::Mat &frame, int frame_idx);
    void waitAndFetchResults();

    const TrackedObjects& getResults() const;

    void PrintPerformanceCounts();
};
