// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

#include <inference_engine.hpp>
#include <ie_common.h>
#include <ie_icnn_network.hpp>
#include <ie_iextension.h>
#include <ie_plugin_config.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include "perf_timer.hpp"
#include "input.hpp"

void loadImageToIEGraph(cv::Mat img, void* ie_buffer);

class VideoFrame;

class IEGraph{
private:
    PerfTimer perfTimerPreprocess;
    PerfTimer perfTimerInfer;

    float confidenceThreshold;

    std::size_t batchSize;

    std::string modelPath;
    std::string cpuExtensionPath;
    std::string cldnnConfigPath;

    std::string inputDataBlobName;
    std::vector<std::string> outputDataBlobNames;

    bool printPerfReport;
    std::string deviceName;

    InferenceEngine::Core ie;
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
    using PostprocessingFunc = std::function<std::vector<Detections>(InferenceEngine::InferRequest::Ptr, const std::vector<std::string>&, cv::Size)>;
    PostprocessingFunc postprocessing;
    using PostLoadFunc = std::function<void (const std::vector<std::string>&, InferenceEngine::CNNNetwork&)>;
    PostLoadFunc postLoad;
    std::thread getterThread;

    void initNetwork(const std::string& deviceName);

public:
    struct InitParams {
        std::size_t batchSize = 1;
        std::size_t maxRequests = 5;
        bool collectStats = false;
        bool reportPerf = false;
        std::string modelPath;
        std::string cpuExtPath;
        std::string cldnnConfigPath;
        std::string deviceName;
        PostLoadFunc postLoadFunc = nullptr;
    };

    explicit IEGraph(const InitParams& p);

    void start(GetterFunc getterFunc, PostprocessingFunc postprocessingFunc);

    bool isRunning();

    InferenceEngine::SizeVector getInputDims() const;

    std::vector<std::shared_ptr<VideoFrame>> getBatchData(cv::Size windowSize);

    unsigned int getBatchSize() const;

    void setDetectionConfidence(float conf);

    ~IEGraph();

    struct Stats {
        float preprocessTime;
        float inferTime;
    };

    Stats getStats() const;

    void printPerformanceCounts(std::string fullDeviceName);
};

