// Copyright (C) 2018-2024 Intel Corporation
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

static inline size_t roundUp(size_t enumerator, size_t denominator) {
    assert(enumerator > 0);
    assert(denominator > 0);
    return 1 + (enumerator - 1) / denominator;
}

static inline std::queue<ov::InferRequest> compile(std::shared_ptr<ov::Model>&& model, const std::string& modelPath,
        const std::string& device, size_t performanceHintNumRequests, ov::Core& core) {
    core.set_property("CPU", ov::affinity(ov::Affinity::NONE));
    ov::CompiledModel compiled = core.compile_model(model, device, {
        {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
        {ov::hint::num_requests(performanceHintNumRequests)}});
    unsigned maxRequests = compiled.get_property(ov::optimal_number_of_infer_requests) + 1;
    logCompiledModelInfo(compiled, modelPath, device);
    slog::info << "\tNumber of network inference requests: " << std::to_string(maxRequests) << slog::endl;
    std::queue<ov::InferRequest> reqQueue;
    for (unsigned i = 0; i < maxRequests; ++i) {
        reqQueue.push(compiled.create_infer_request());
    }
    return reqQueue;
}

class IEGraph{
private:
    PerfTimer perfTimerPreprocess;
    PerfTimer perfTimerInfer;
    std::queue<ov::InferRequest> availableRequests;

    struct BatchRequestDesc {
        std::vector<std::shared_ptr<VideoFrame>> vfPtrVec;
        ov::InferRequest req;
        std::chrono::high_resolution_clock::time_point startTime;
    };
    std::queue<BatchRequestDesc> busyBatchRequests;

    std::size_t maxRequests;

    std::atomic_bool terminate = {false};
    std::mutex mtxAvalableRequests;
    std::mutex mtxBusyRequests;
    std::condition_variable condVarAvailableRequests;
    std::condition_variable condVarBusyRequests;

    using GetterFunc = std::function<bool(VideoFrame&)>;
    GetterFunc getter;
    using PostprocessingFunc = std::function<std::vector<Detections>(ov::InferRequest, cv::Size)>;
    PostprocessingFunc postprocessing;
    std::thread getterThread;
public:
    IEGraph(std::queue<ov::InferRequest>&& availableRequests, bool collectStats):
        perfTimerPreprocess(collectStats ? PerfTimer::DefaultIterationsCount : 0),
        perfTimerInfer(collectStats ? PerfTimer::DefaultIterationsCount : 0),
        availableRequests(std::move(availableRequests)),
        maxRequests(this->availableRequests.size()) {}

    void start(size_t batchSize, GetterFunc getterFunc, PostprocessingFunc postprocessingFunc);

    bool isRunning();

    std::vector<std::shared_ptr<VideoFrame>> getBatchData(cv::Size windowSize);

    ~IEGraph();

    struct Stats {
        float preprocessTime;
        float inferTime;
    };

    Stats getStats() const;
};
