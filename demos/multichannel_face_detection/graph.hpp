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

#include <vector>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>
#include <atomic>
#include <string>
#include <memory>

#include "opencv2/opencv.hpp"

#include <inference_engine.hpp>
#include <ie_common.h>
#include <ie_plugin_ptr.hpp>
#include <ie_icnn_network.hpp>
#include <mkldnn/mkldnn_extension.hpp>
#include <mkldnn/mkldnn_extension_ptr.hpp>
#include <ie_extension.h>
#include <cpp/ie_cnn_net_reader.h>
#include <ie_plugin_config.hpp>

#include "perf_timer.hpp"

#include "input.hpp"

void loadImageToIEGraph(cv::Mat img, void* ie_buffer);

struct Face{
    cv::Rect2f rect;
    float confidence;
    unsigned char age;
    unsigned char gender;
    Face(cv::Rect2f r, float c, unsigned char a, unsigned char g): rect(r), confidence(c), age(a), gender(g) {}
};

class VideoFrame;

class IEGraph{
private:
    PerfTimer perf_timer_preprocess;
    PerfTimer perf_timer_infer;

    float confidenceThreshold;

    unsigned int batchSize;

    std::string modelPath;
    std::string weightsPath;
    std::string cpuExtensionPath;
    std::string cldnnConfigPath;

    std::string inputDataBlobName;
    std::string outputDataBlobName;

    InferenceEngine::InferencePlugin plugin;
    std::queue<InferenceEngine::InferRequest::Ptr> availableRequests;

    struct BatchRequestDesc {
        std::vector<std::shared_ptr<VideoFrame>> vf_ptr_vec;
        InferenceEngine::InferRequest::Ptr req;
        std::chrono::high_resolution_clock::time_point start_time;
    };
    std::queue<BatchRequestDesc> busyBatchRequests;

    std::atomic_bool terminate = {false};
    std::mutex mutexPush;
    std::mutex mutexPop;

    using GetterFunc = std::function<bool(VideoFrame&)>;
    GetterFunc getter;

    std::thread getterThread;
    std::condition_variable condVarPush;
    std::condition_variable condVarPop;

    void start(GetterFunc _getter);

    void initNetwork(size_t max_requests_, const std::string& device_name);

public:
    IEGraph(bool collect_stats, size_t max_requests_, unsigned int bs,
            const std::string& model,       const std::string& weights,
            const std::string& cpu_ext,     const std::string& cl_ext,
            const std::string& device_name, GetterFunc _getter);

    std::vector<std::shared_ptr<VideoFrame>> getBatchData();

    unsigned int getBatchSize() const {
        return batchSize;
    }

    void setDetectionConfidence(float  conf) {
        confidenceThreshold = conf;
    }

    ~IEGraph();

    struct Stats {
        float preprocess_time;
        float infer_time;
    };

    Stats getStats() const;
};

