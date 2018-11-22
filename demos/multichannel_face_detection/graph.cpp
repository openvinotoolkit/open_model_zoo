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
#include <string>
#include <memory>
#include <vector>
#include <utility>
#include <algorithm>

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
    InferenceEngine::CNNNetReader  netReader;

    netReader.ReadNetwork(modelPath);
    netReader.ReadWeights(weightsPath);

    if (!netReader.isParseSuccess()) {
        throw std::logic_error("Failed to parse model!");
    }

    plugin = InferenceEngine::PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(deviceName);
    plugin.AddExtension(std::make_shared<InferenceEngine::Extensions::Cpu::CpuExtensions>());
    if (deviceName.find("CPU") != std::string::npos) {
        plugin.SetConfig({{InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, "NO"}});
    }
    if (!cpuExtensionPath.empty()) {
        auto extension_ptr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(cpuExtensionPath);
        plugin.AddExtension(extension_ptr);
    }
    if (!cldnnConfigPath.empty()) {
        plugin.SetConfig({{InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, cldnnConfigPath}});
    }

    // Set batch size
    if (batchSize > 1) {
        auto inShapes = netReader.getNetwork().getInputShapes();
        for (auto& pair : inShapes) {
            auto& dims = pair.second;
            if (!dims.empty()) {
                dims[0] = batchSize;
            }
        }
        netReader.getNetwork().reshape(inShapes);
    }

    InferenceEngine::ExecutableNetwork network;
    network = plugin.LoadNetwork(netReader.getNetwork(), {});

    InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error("Face Detection network should have only one input");
    }
    inputDataBlobName = inputInfo.begin()->first;

    InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
    if (outputInfo.size() != 1) {
        throw std::logic_error("Face Detection network should have only one output");
    }
    outputDataBlobName = outputInfo.begin()->first;

    for (size_t i = 0; i < maxRequests; ++i) {
        auto req = network.CreateInferRequestPtr();
        availableRequests.push(req);
    }

    availableRequests.front()->StartAsync();
    availableRequests.front()->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
}

void IEGraph::start(GetterFunc getterFunc) {
    assert(nullptr != getterFunc);
    assert(nullptr == getter);
    getter = std::move(getterFunc);
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
                    if (terminate) {
                        break;
                    }
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
    });
}

IEGraph::IEGraph(const InitParams& p):
    perfTimerPreprocess(p.collectStats ? PerfTimer::DefaultIterationsCount : 0),
    perfTimerInfer(p.collectStats ? PerfTimer::DefaultIterationsCount : 0),
    confidenceThreshold(0.5f), batchSize(p.batchSize),
    modelPath(p.modelPath), weightsPath(p.weightsPath),
    cpuExtensionPath(p.cpuExtPath), cldnnConfigPath(p.cldnnConfigPath),
    maxRequests(p.maxRequests) {
    assert(p.maxRequests > 0);

    initNetwork(p.deviceName);
}

InferenceEngine::SizeVector IEGraph::getInputDims() const {
    assert(!availableRequests.empty());
    auto inputBlob = availableRequests.front()->GetBlob(inputDataBlobName);
    return inputBlob->getTensorDesc().getDims();
}

std::vector<std::shared_ptr<VideoFrame> > IEGraph::getBatchData() {
    std::vector<std::shared_ptr<VideoFrame>> vframes;
    InferenceEngine::InferRequest::Ptr req;
    std::chrono::high_resolution_clock::time_point startTime;
    {
        std::unique_lock<std::mutex> lock(mtxBusyRequests);
        condVarBusyRequests.wait(lock, [&]() {
            return !busyBatchRequests.empty();
        });
        vframes = std::move(busyBatchRequests.front().vfPtrVec);
        req = std::move(busyBatchRequests.front().req);
        startTime = std::move(busyBatchRequests.front().startTime);
        busyBatchRequests.pop();
    }

    if (nullptr != req && InferenceEngine::OK == req->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY)) {
        auto output = req->GetBlob(outputDataBlobName);

        float* dataPtr = output->buffer();
        InferenceEngine::SizeVector svec = output->dims();
        size_t total = 1;
        for (size_t j = 0; j < svec.size(); j++) {
            total *= svec[j];
        }

        for (size_t b = 0 ; b < vframes.size(); b++) {
            vframes[b]->detections.clear();
        }

        for (size_t i = 0; i < total; i+=7) {
            float conf = dataPtr[i + 2];
            if (conf > confidenceThreshold) {
                int idxInBatch = static_cast<int>(dataPtr[i]);
                float x0 = std::min(std::max(0.0f, dataPtr[i + 3]), 1.0f);
                float y0 = std::min(std::max(0.0f, dataPtr[i + 4]), 1.0f);
                float x1 = std::min(std::max(0.0f, dataPtr[i + 5]), 1.0f);
                float y1 = std::min(std::max(0.0f, dataPtr[i + 6]), 1.0f);

                cv::Rect2f rect = {x0 , y0, x1-x0, y1-y0};
                vframes[idxInBatch]->detections.push_back(Face(rect, conf, 0, 0));
            }
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
        slog::info << "Performance counts report" << slog::endl << slog::endl;
        printPerformanceCounts();
        condVarAvailableRequests.notify_one();
    }
    if (getterThread.joinable()) {
        getterThread.join();
    }
}

IEGraph::Stats IEGraph::getStats() const {
    return Stats{perfTimerPreprocess.getValue(), perfTimerInfer.getValue()};
}

void IEGraph::printPerformanceCounts() {
    ::printPerformanceCounts(availableRequests.front()->GetPerformanceCounts(), std::cout, false);
}
