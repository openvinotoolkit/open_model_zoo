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

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#endif

namespace {

void loadImgToIEGraph(const cv::Mat& img, size_t batch, void* ieBuffer) {
    const int channels = img.channels();
    const int height = img.rows;
    const int width = img.cols;

    float* ieData = reinterpret_cast<float*>(ieBuffer);
    int bOffset = static_cast<int>(batch) * channels * width * height;
    for (int c = 0; c < channels; c++) {
        int cOffset = c * width * height;
        for (int w = 0; w < width; w++) {
            for (int h = 0; h < height; h++) {
                ieData[bOffset + cOffset + h * width + w] =
                        static_cast<float>(img.at<cv::Vec3b>(h, w)[c]);
            }
        }
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
        const ov::Shape input_shape = availableRequests.front().get_input_tensor().get_shape();
        std::vector<cv::Mat> imgsToProc(batchSize, cv::Mat(inSize, CV_8UC3));
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

            auto preprocess = [&]() {
                float* inputPtr = req.get_input_tensor().data<float>();
                auto loopBody = [&](size_t i) {
                    cv::resize(vframes[i]->frame,
                               imgsToProc[i],
                               imgsToProc[i].size());
                    loadImgToIEGraph(imgsToProc[i], i, inputPtr);
                };
#ifdef USE_TBB
                run_in_arena([&](){
                    tbb::parallel_for<size_t>(0, batchSize, loopBody);
                });
#else
                for (size_t i = 0; i < batchSize; i++) {
                    loopBody(i);
                }
#endif
            };

            if (perfTimerInfer.enabled()) {
                {
                    ScopedTimer st(perfTimerPreprocess);
                    preprocess();
                }
                auto startTime = std::chrono::high_resolution_clock::now();
                req.start_async();
                std::unique_lock<std::mutex> lock(mtxBusyRequests);
                busyBatchRequests.push({std::move(vframes), std::move(req), startTime});
            } else {
                preprocess();
                req.start_async();
                std::unique_lock<std::mutex> lock(mtxBusyRequests);
                busyBatchRequests.push({std::move(vframes), std::move(req),
                                    std::chrono::high_resolution_clock::time_point()});
            }
            condVarBusyRequests.notify_one();
        }
        condVarBusyRequests.notify_one(); // notify that there will be no new InferRequests
    });
}

IEGraph::IEGraph(const InitParams& p):
        perfTimerPreprocess(p.collectStats ? PerfTimer::DefaultIterationsCount : 0),
        perfTimerInfer(p.collectStats ? PerfTimer::DefaultIterationsCount : 0),
        confidenceThreshold(0.5f), batchSize(p.batchSize),
        modelPath(p.modelPath),
        postRead(p.postReadFunc) {
    std::shared_ptr<ov::Function> model = core.read_model(modelPath);
    if (model->get_parameters().size() != 1) {
        throw std::logic_error("Face Detection model must have only one input");
    }
    const ov::Layout inLyout{"NCHW"};
    model = ov::preprocess::PrePostProcessor().input(ov::preprocess::InputInfo().tensor(ov::preprocess::InputTensorInfo().set_layout(inLyout))).build(model);
    ov::Shape inShape = model->input().get_shape();
    inSize = {int(inShape[ov::layout::width_idx(inLyout)]), int(inShape[ov::layout::height_idx(inLyout)])};
    // Set batch size
    inShape[ov::layout::batch_idx(inLyout)] = batchSize;
    model->reshape({{model->input().get_any_name(), inShape}});

    if (postRead != nullptr)
        postRead(model);
    core.set_config({{"CPU_BIND_THREAD", "NO"}}, "CPU");
    ov::runtime::ExecutableNetwork net = core.compile_model(model, p.deviceName, {{"PERFORMANCE_HINT", "THROUGHPUT"}});
    maxRequests = net.get_metric("OPTIMAL_NUMBER_OF_INFER_REQUESTS").as<unsigned>() + 1;
    logExecNetworkInfo(net, modelPath, p.deviceName);

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

unsigned int IEGraph::getBatchSize() const {
    return static_cast<unsigned int>(batchSize);
}

void IEGraph::setDetectionConfidence(float conf) {
    confidenceThreshold = conf;
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
