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

#include <openvino/openvino.hpp>

#include <utils/common.hpp>
#include <utils/slog.hpp>
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
    ov::runtime::Core core;
    std::queue<ov::runtime::InferRequest> availableRequests;

    struct BatchRequestDesc {
        std::vector<std::shared_ptr<VideoFrame>> vfPtrVec;
        ov::runtime::InferRequest req;
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
    using PostprocessingFunc = std::function<std::vector<Detections>(ov::runtime::InferRequest, cv::Size)>;
    PostprocessingFunc postprocessing;
    using PostReadFunc = std::function<void (std::shared_ptr<ov::Function>)>;
    PostReadFunc postRead;
    std::thread getterThread;
public:
    struct InitParams {
        std::size_t batchSize = 1;
        bool collectStats = false;
        std::string modelPath;
        std::string deviceName;
        PostReadFunc postReadFunc = nullptr;
    };

    explicit IEGraph(const InitParams& p);

    void start(GetterFunc getterFunc, PostprocessingFunc postprocessingFunc);

    bool isRunning();

    ov::Shape getInputShape();

    std::vector<std::shared_ptr<VideoFrame>> getBatchData(cv::Size windowSize);

    unsigned int getBatchSize() const;

    void setDetectionConfidence(float conf);

    ~IEGraph();

    struct Stats {
        float preprocessTime;
        float inferTime;
    };

    Stats getStats() const;
};
