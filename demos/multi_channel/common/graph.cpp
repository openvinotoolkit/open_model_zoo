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

void IEGraph::initNetwork(const std::string& deviceName) {
    auto cnnNetwork = ie.ReadNetwork(modelPath);

    if (deviceName.find("CPU") != std::string::npos) {
        ie.SetConfig({{InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, "NO"}}, "CPU");
    }
    if (!cpuExtensionPath.empty()) {
        auto extension_ptr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(cpuExtensionPath);
        ie.AddExtension(extension_ptr, "CPU");
    }
    if (!cldnnConfigPath.empty()) {
        ie.SetConfig({{InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, cldnnConfigPath}}, "GPU");
    }
    /** Setting parameter for collecting per layer metrics **/
    if (printPerfReport) {
        ie.SetConfig({ { InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES } });
    }

    // Set batch size
    if (batchSize > 1) {
        auto inShapes = cnnNetwork.getInputShapes();
        for (auto& pair : inShapes) {
            auto& dims = pair.second;
            if (!dims.empty()) {
                dims[0] = batchSize;
            }
        }
        cnnNetwork.reshape(inShapes);
    }

    InferenceEngine::ExecutableNetwork network;
    network = ie.LoadNetwork(cnnNetwork, deviceName);

    InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error("Face Detection network should have only one input");
    }
    inputDataBlobName = inputInfo.begin()->first;

    InferenceEngine::OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
    outputDataBlobNames.reserve(outputInfo.size());
    for (const auto& i : outputInfo) {
        outputDataBlobNames.push_back(i.first);
    }

    for (size_t i = 0; i < maxRequests; ++i) {
        auto req = network.CreateInferRequestPtr();
        availableRequests.push(req);
    }

    if (postLoad != nullptr)
        postLoad(outputDataBlobNames, cnnNetwork);

    availableRequests.front()->StartAsync();
    availableRequests.front()->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
}

void IEGraph::start(GetterFunc getterFunc, PostprocessingFunc postprocessingFunc) {
    assert(nullptr != getterFunc);
    assert(nullptr != postprocessingFunc);
    assert(nullptr == getter);
    getter = std::move(getterFunc);
    postprocessing = std::move(postprocessingFunc);
    getterThread = std::thread([&]() {
        std::vector<std::shared_ptr<VideoFrame>> vframes;
        std::vector<cv::Mat> imgsToProc(batchSize);
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

            InferenceEngine::InferRequest::Ptr req;
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

            auto inputBlob = req->GetBlob(inputDataBlobName);
            imgsToProc.resize(batchSize);
            for (size_t i = 0; i < batchSize; i++) {
                if (imgsToProc[i].empty()) {
                    auto& dims = inputBlob->getTensorDesc().getDims();
                    assert(4 == dims.size());
                    auto height = static_cast<int>(dims[2]);
                    auto width  = static_cast<int>(dims[3]);
                    imgsToProc[i] = cv::Mat(height, width, CV_8UC3);
                }
            }

            auto preprocess = [&]() {
                auto buff = inputBlob->buffer();
                float* inputPtr = static_cast<float*>(buff);
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
                req->StartAsync();
                std::unique_lock<std::mutex> lock(mtxBusyRequests);
                busyBatchRequests.push({std::move(vframes), std::move(req), startTime});
            } else {
                preprocess();
                req->StartAsync();
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
    cpuExtensionPath(p.cpuExtPath), cldnnConfigPath(p.cldnnConfigPath),
    printPerfReport(p.reportPerf), deviceName(p.deviceName),
    maxRequests(p.maxRequests) {
    assert(p.maxRequests > 0);

    postLoad = p.postLoadFunc;
    initNetwork(p.deviceName);
}

bool IEGraph::isRunning() {
    std::lock_guard<std::mutex> lock(mtxBusyRequests);
    return !terminate || !busyBatchRequests.empty();
}

InferenceEngine::SizeVector IEGraph::getInputDims() const {
    assert(!availableRequests.empty());
    auto inputBlob = availableRequests.front()->GetBlob(inputDataBlobName);
    return inputBlob->getTensorDesc().getDims();
}

std::vector<std::shared_ptr<VideoFrame> > IEGraph::getBatchData(cv::Size frameSize) {
    std::vector<std::shared_ptr<VideoFrame>> vframes;
    InferenceEngine::InferRequest::Ptr req;
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

    if (nullptr != req && InferenceEngine::OK == req->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY)) {
        auto detections = postprocessing(req, outputDataBlobNames, frameSize);
        for (decltype(detections.size()) i = 0; i < detections.size(); i ++) {
            vframes[i]->detections = std::move(detections[i]);
        }
        if (perfTimerInfer.enabled()) {
            auto endTime = std::chrono::high_resolution_clock::now();
            perfTimerInfer.addValue(endTime - startTime);
        }
    }

    if (nullptr != req) {
        std::unique_lock<std::mutex> lock(mtxAvalableRequests);
        availableRequests.push(std::move(req));
        lock.unlock();
        condVarAvailableRequests.notify_one();
    }

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
        bool ready = false;
        while (!ready) {
            std::unique_lock<std::mutex> lock(mtxBusyRequests);
            if (!busyBatchRequests.empty()) {
                auto& req = busyBatchRequests.front().req;
                if (nullptr != req) {
                    req->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
                    availableRequests.push(std::move(req));
                }
                busyBatchRequests.pop();
            }
            if (availableRequests.size() == maxRequests) {
                ready = true;
            }
        }
        if (printPerfReport) {
            slog::info << "Performance counts report" << slog::endl << slog::endl;
            printPerformanceCounts(getFullDeviceName(ie, deviceName));
        }
        condVarAvailableRequests.notify_one();
    }
    if (getterThread.joinable()) {
        getterThread.join();
    }
}

IEGraph::Stats IEGraph::getStats() const {
    return Stats{perfTimerPreprocess.getValue(), perfTimerInfer.getValue()};
}

void IEGraph::printPerformanceCounts(std::string fullDeviceName) {
    ::printPerformanceCounts(*availableRequests.front(), std::cout, fullDeviceName, false);
}
