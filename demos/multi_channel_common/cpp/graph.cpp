// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "graph.hpp"
#include "threading.hpp"

namespace {
void framesToTensor(const std::vector<std::shared_ptr<VideoFrame>>& frames, ov::runtime::Tensor tensor) {
    static const ov::Layout layout{"NHWC"};
    static const ov::Shape shape = tensor.get_shape();
    static const size_t batchSize = shape[ov::layout::batch_idx(layout)];
    static const cv::Size inSize{int(shape[ov::layout::width_idx(layout)]), int(shape[ov::layout::height_idx(layout)])};
    static const size_t channels = shape[ov::layout::channels_idx(layout)];
    static const size_t batchOffset = inSize.area() * channels;
    assert(batchSize == frames.size()]);
    assert(channels == 3);
    uint8_t* data = tensor.data<uint8_t>();
    for (size_t i = 0; i < batchSize; ++i) {
        assert(frames[i]->frame.channels() == channels);
        cv::resize(frames[i]->frame, cv::Mat{inSize, CV_8UC3, static_cast<void*>(data + batchOffset * i)}, inSize);
    }
}
}  // namespace

void IEGraph::start(GetterFunc getterFunc, PostprocessingFunc postprocessingFunc) {
    assert(nullptr != getterFunc);
    assert(nullptr != postprocessingFunc);
    assert(nullptr == getter);
    getter = std::move(getterFunc);
    postprocessing = std::move(postprocessingFunc);
    getterThread = std::thread([&]() {
        std::vector<std::shared_ptr<VideoFrame>> vframes;
        while (!terminate) {
            vframes.clear();
            size_t b = 0;
            while (b != batchSize) {
                VideoFrame vframe;
                if (getter(vframe)) {
                    vframes.push_back(std::make_shared<VideoFrame>(vframe));
                    ++b;
                } else {
                    terminate = true;
                    break;
                }
            }

            ov::runtime::InferRequest req;
            {
                std::unique_lock<std::mutex> lock(mtxAvalableRequests);
                condVarAvailableRequests.wait(lock, [&]() {
                    return !availableRequests.empty() || terminate;
                });
                if (terminate) {
                    break;
                }
                req = std::move(availableRequests.front());
                availableRequests.pop();
            }

            if (perfTimerInfer.enabled()) {
                {
                    ScopedTimer st(perfTimerPreprocess);
                    framesToTensor(vframes, req.get_input_tensor());
                }
                auto startTime = std::chrono::high_resolution_clock::now();
                req.start_async();
                std::unique_lock<std::mutex> lock(mtxBusyRequests);
                busyBatchRequests.push({std::move(vframes), std::move(req), startTime});
            } else {
                framesToTensor(vframes, req.get_input_tensor());
                req.start_async();
                std::unique_lock<std::mutex> lock(mtxBusyRequests);
                busyBatchRequests.push({std::move(vframes), std::move(req),
                                    std::chrono::high_resolution_clock::time_point()});
            }
            condVarBusyRequests.notify_one();
        }
        condVarBusyRequests.notify_one();  // notify that there will be no new InferRequests
    });
}

IEGraph::IEGraph(const std::string& modelPath, const std::string& device, ov::runtime::Core& core,
            bool collectStats, std::size_t batchSize, PostReadFunc&& postReadFunc):
        perfTimerPreprocess(collectStats ? PerfTimer::DefaultIterationsCount : 0),
        perfTimerInfer(collectStats ? PerfTimer::DefaultIterationsCount : 0),
        batchSize(batchSize),
        postRead(std::move(postReadFunc)) {
    std::shared_ptr<ov::Function> model = core.read_model(modelPath);
    if (model->get_parameters().size() != 1) {
        throw std::logic_error("Face Detection model must have only one input");
    }
    const ov::Layout inLyout{"NHWC"};
    model = ov::preprocess::PrePostProcessor(model).input(ov::preprocess::InputInfo()
        .tensor(ov::preprocess::InputTensorInfo()
            .set_element_type(ov::element::u8)
            .set_layout(inLyout))
        .preprocess(ov::preprocess::PreProcessSteps()
            .convert_element_type(ov::element::f32)
            .convert_layout("NCHW"))
        .network(ov::preprocess::InputNetworkInfo().set_layout("NCHW"))
    ).build();
    ov::Shape inShape = model->input().get_shape();
    // Set batch size
    inShape[ov::layout::batch_idx(inLyout)] = batchSize;
    model->reshape({{model->input().get_any_name(), inShape}});

    if (postRead != nullptr)
        postRead(model);
    core.set_config({{"CPU_BIND_THREAD", "NO"}}, "CPU");
    ov::runtime::ExecutableNetwork net = core.compile_model(model, device, {{"PERFORMANCE_HINT", "THROUGHPUT"}});
    maxRequests = net.get_metric("OPTIMAL_NUMBER_OF_INFER_REQUESTS").as<unsigned>() + 1;
    logExecNetworkInfo(net, modelPath, device);

    slog::info << "\tNumber of network inference requests: " << maxRequests << slog::endl;
    slog::info << "\tBatch size is set to " << batchSize << slog::endl;

    for (size_t i = 0; i < maxRequests; ++i) {
        availableRequests.push(net.create_infer_request());
    }
}

bool IEGraph::isRunning() {
    std::lock_guard<std::mutex> lock(mtxBusyRequests);
    return !terminate || !busyBatchRequests.empty();
}

ov::Shape IEGraph::getInputShape() {
    return availableRequests.front().get_input_tensor().get_shape();
}

std::vector<std::shared_ptr<VideoFrame>> IEGraph::getBatchData(cv::Size frameSize) {
    std::vector<std::shared_ptr<VideoFrame>> vframes;
    ov::runtime::InferRequest req;
    std::chrono::high_resolution_clock::time_point startTime;
    {
        std::unique_lock<std::mutex> lock(mtxBusyRequests);
        condVarBusyRequests.wait(lock, [&]() {
            // wait until the pipeline is stopped or there are new InferRequests
            return terminate || !busyBatchRequests.empty();
        });
        if (busyBatchRequests.empty()) {
            return {}; // woke up because of termination, so leave if nothing to preces
        }
        vframes = std::move(busyBatchRequests.front().vfPtrVec);
        req = std::move(busyBatchRequests.front().req);
        startTime = std::move(busyBatchRequests.front().startTime);
        busyBatchRequests.pop();
    }

    req.wait();
    auto detections = postprocessing(req, frameSize);
    for (decltype(detections.size()) i = 0; i < detections.size(); i ++) {
        vframes[i]->detections = std::move(detections[i]);
    }
    if (perfTimerInfer.enabled()) {
        auto endTime = std::chrono::high_resolution_clock::now();
        perfTimerInfer.addValue(endTime - startTime);
    }

    {
        std::unique_lock<std::mutex> lock(mtxAvalableRequests);
        availableRequests.push(std::move(req));
    }
    condVarAvailableRequests.notify_one();

    return vframes;
}

IEGraph::~IEGraph() {
    terminate = true;
    {
        std::unique_lock<std::mutex> lock(mtxAvalableRequests);
        while (availableRequests.size() != maxRequests) {
            std::unique_lock<std::mutex> lock(mtxBusyRequests);
            if (!busyBatchRequests.empty()) {
                auto& req = busyBatchRequests.front().req;
                req.cancel();
                availableRequests.push(std::move(req));
                busyBatchRequests.pop();
            }
        }
    }
    condVarAvailableRequests.notify_one();
    if (getterThread.joinable()) {
        getterThread.join();
    }
}

IEGraph::Stats IEGraph::getStats() const {
    return Stats{perfTimerPreprocess.getValue(), perfTimerInfer.getValue()};
}
