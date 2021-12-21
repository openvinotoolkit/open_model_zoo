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

namespace {
constexpr size_t roundUp(size_t enumerator, size_t denominator) {
    assert(enumerator > 0);
    assert(denominator > 0);
    return 1 + (enumerator - 1) / denominator;
}

std::shared_ptr<ov::Model> setBatch(std::shared_ptr<ov::Model>&& model, size_t batchSize) {
    if (model->get_parameters().size() != 1) {
        throw std::logic_error("Face Detection model must have only one input");
    }
    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC");
    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();
    ov::set_batch(model, batchSize);
    return model;
}

std::queue<ov::runtime::InferRequest> setConfig(std::shared_ptr<ov::Model>&& model, const std::string& modelPath,
        const std::string& device, size_t performanceHintNumRequests, ov::runtime::Core& core) {
    core.set_config({{"CPU_BIND_THREAD", "NO"}}, "CPU");
    ov::runtime::CompiledModel compiled = core.compile_model(model, device, {
        {"PERFORMANCE_HINT", "THROUGHPUT"},
        {"PERFORMANCE_HINT_NUM_REQUESTS", std::to_string(performanceHintNumRequests)}});
    unsigned maxRequests = compiled.get_metric("OPTIMAL_NUMBER_OF_INFER_REQUESTS").as<unsigned>() + 1;
    logExecNetworkInfo(compiled, modelPath, device);
    slog::info << "\tNumber of network inference requests: " << maxRequests << slog::endl;
    std::queue<ov::runtime::InferRequest> reqQueue;
    for (unsigned i = 0; i < maxRequests; ++i) {
        reqQueue.push(compiled.create_infer_request());
    }
    return reqQueue;
}
}  // namespace

class VideoFrame;

class IEGraph{
private:
    PerfTimer perfTimerPreprocess;
    PerfTimer perfTimerInfer;
    std::queue<ov::runtime::InferRequest> availableRequests;

    struct BatchRequestDesc {
        std::vector<std::shared_ptr<VideoFrame>> vfPtrVec;
        ov::runtime::InferRequest req;
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
    using PostprocessingFunc = std::function<std::vector<Detections>(ov::runtime::InferRequest, cv::Size)>;
    PostprocessingFunc postprocessing;
    std::thread getterThread;
public:
    IEGraph::IEGraph(std::queue<ov::runtime::InferRequest>&& availableRequests, bool collectStats):
        availableRequests(std::move(availableRequests)),
        maxRequests(this->availableRequests.size()),
        perfTimerPreprocess(collectStats ? PerfTimer::DefaultIterationsCount : 0),
        perfTimerInfer(collectStats ? PerfTimer::DefaultIterationsCount : 0) {}

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
