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
#include <ie_iextension.h>
#include <cpp/ie_cnn_net_reader.h>
#include <ie_plugin_config.hpp>

#include <samples/slog.hpp>
#include <samples/common.hpp>
#include "perf_timer.hpp"
#include "input.hpp"
#include <ext_list.hpp>

void loadImageToIEGraph(cv::Mat img, void* ie_buffer);

class VideoFrame;

class IEGraph{
private:
    PerfTimer perfTimerPreprocess;
    PerfTimer perfTimerInfer;

    float confidenceThreshold;

    std::size_t batchSize;

    std::string modelPath;
    std::string weightsPath;
    std::string cpuExtensionPath;
    std::string cldnnConfigPath;

    std::string inputDataBlobName;
    std::string outputDataBlobName;

    bool printPerfReport = false;

    InferenceEngine::InferencePlugin plugin;
    std::queue<InferenceEngine::InferRequest::Ptr> availableRequests;

    struct BatchRequestDesc {
        std::vector<std::shared_ptr<VideoFrame>> vfPtrVec;
        InferenceEngine::InferRequest::Ptr req;
        std::chrono::high_resolution_clock::time_point startTime;
    };
    std::queue<BatchRequestDesc> busyBatchRequests;

    std::size_t maxRequests = 0;

    std::atomic_bool terminate = {false};
    std::mutex mtxAvalableRequests;
    std::mutex mtxBusyRequests;
    std::condition_variable condVarAvailableRequests;
    std::condition_variable condVarBusyRequests;

    using GetterFunc = std::function<bool(VideoFrame&)>;
    GetterFunc getter;
    std::thread getterThread;

    void initNetwork(const std::string& deviceName);

public:
    struct InitParams {
        std::size_t batchSize = 1;
        std::size_t maxRequests = 5;
        bool collectStats = false;
        bool reportPerf = false;
        std::string modelPath;
        std::string weightsPath;
        std::string cpuExtPath;
        std::string cldnnConfigPath;
        std::string deviceName;
    };

    explicit IEGraph(const InitParams& p);

    void start(GetterFunc getterFunc);

    InferenceEngine::SizeVector getInputDims() const;

    std::vector<std::shared_ptr<VideoFrame>> getBatchData();

    unsigned int getBatchSize() const;

    void setDetectionConfidence(float conf);

    ~IEGraph();

    struct Stats {
        float preprocessTime;
        float inferTime;
    };

    Stats getStats() const;

    void printPerformanceCounts();
};

