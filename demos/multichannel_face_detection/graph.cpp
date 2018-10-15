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

void loadImagesToIEGraph(const std::vector<cv::Mat>& imgs, void* ie_buffer) {
    const int batch = imgs.size();
    if (!batch)
        return;

    const int channels = imgs[0].channels();
    const int height = imgs[0].rows;
    const int width = imgs[0].cols;

    float* ie_data = reinterpret_cast<float*>(ie_buffer);
    for (int b = 0; b < batch; b++) {
        int b_offset = b * channels * width * height;
        for (int c = 0; c < channels; c++) {
            int c_offset = c * width * height;
            for (int w = 0; w < width; w++) {
                for (int h = 0; h < height; h++) {
                    ie_data[b_offset + c_offset + h * width + w] =
                            static_cast<float>(imgs[b].at<cv::Vec3b>(h, w)[c]);
                }
            }
        }
    }
}

void IEGraph::initNetwork(size_t max_requests_, const std::string& device_name) {
    InferenceEngine::CNNNetReader  netReader;

    netReader.ReadNetwork(modelPath);
    netReader.ReadWeights(weightsPath);

    if (!netReader.isParseSuccess()) {
        throw std::logic_error("Failed to parse model!");
    }

    plugin = InferenceEngine::PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(device_name);
    if (device_name.find("CPU") != std::string::npos) {
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
        netReader.getNetwork().setBatchSize(batchSize);
    }

    InferenceEngine::ExecutableNetwork network;
    network = plugin.LoadNetwork(netReader.getNetwork(), {});

    InferenceEngine::InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error("Face Detection network should have only one input");
    }
    inputDataBlobName  = inputInfo.begin()->first;

    InferenceEngine::OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
    if (outputInfo.size() != 1) {
        throw std::logic_error("Face Detection network should have only one output");
    }
    outputDataBlobName = outputInfo.begin()->first;

    for (size_t i = 0; i < max_requests_; ++i) {
        auto req = network.CreateInferRequestPtr();
        availableRequests.push(req);
    }

    availableRequests.front()->StartAsync();
    availableRequests.front()->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
}

void IEGraph::start(GetterFunc _getter) {
    assert(nullptr != _getter);
    getter = std::move(_getter);
    getterThread = std::thread([&]() {
        std::vector<std::shared_ptr<VideoFrame>> vframes;
        std::vector<cv::Mat> imgs_to_proc(batchSize);
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
                std::unique_lock<std::mutex> lock(mutexPush);
                condVarPush.wait(lock, [&]() {
                    return !availableRequests.empty() || terminate;
                });
                if (terminate) {
                    break;
                }
                req = std::move(availableRequests.front());
                availableRequests.pop();
            }

            auto inputBlob = req->GetBlob(inputDataBlobName);
            imgs_to_proc.clear();
            imgs_to_proc.resize(batchSize);
            for (size_t i = 0; i < batchSize; i++) {
                if (imgs_to_proc[i].empty()) {
                    imgs_to_proc[i] = cv::Mat(inputBlob->dims()[1], inputBlob->dims()[0], CV_8UC3);
                }
            }

            auto preprocess = [&]() {
                for (size_t i = 0; i < batchSize; i++) {
                    cv::resize(*(vframes[i]->frame), imgs_to_proc[i], imgs_to_proc[i].size());
                }
                float* input_ptr = static_cast<float*>(inputBlob->buffer());
                loadImagesToIEGraph(imgs_to_proc,
                                   static_cast<void*>(input_ptr));
            };

            if (perf_timer_infer.enabled()) {
                {
                    ScopedTimer st(perf_timer_preprocess);
                    preprocess();
                }
                auto start_time = std::chrono::high_resolution_clock::now();
                req->StartAsync();
                std::unique_lock<std::mutex> lock(mutexPop);
                busyBatchRequests.push({std::move(vframes), std::move(req), start_time});
            } else {
                preprocess();
                req->StartAsync();
                std::unique_lock<std::mutex> lock(mutexPop);
                busyBatchRequests.push({std::move(vframes), std::move(req),
                                    std::chrono::high_resolution_clock::time_point()});
            }
            condVarPop.notify_one();
        }
    });
}

IEGraph::IEGraph(bool collect_stats, size_t max_requests_, unsigned int bs,
                 const std::string& model, const std::string& weights,
                 const std::string& cpu_ext, const std::string& cl_ext,
                 const std::string& device_name, GetterFunc _getter):
    perf_timer_preprocess(collect_stats ? 50 : 0),
    perf_timer_infer(collect_stats ? 50 : 0),
    confidenceThreshold(0.5f), batchSize(bs),
    modelPath(model), weightsPath(weights),
    cpuExtensionPath(cpu_ext), cldnnConfigPath(cl_ext) {
    assert(max_requests_ > 0);

    initNetwork(max_requests_, device_name);
    start(std::move(_getter));
}

std::vector<std::shared_ptr<VideoFrame> > IEGraph::getBatchData() {
    std::vector<std::shared_ptr<VideoFrame>> vframes;
    size_t count = 0;
    InferenceEngine::InferRequest::Ptr req;
    std::chrono::high_resolution_clock::time_point start_time;
    {
        std::unique_lock<std::mutex> lock(mutexPop);
        condVarPop.wait(lock, [&]() {
            return !busyBatchRequests.empty();
        });
        vframes = std::move(busyBatchRequests.front().vf_ptr_vec);
        req = std::move(busyBatchRequests.front().req);
        start_time = std::move(busyBatchRequests.front().start_time);
        busyBatchRequests.pop();
        count = busyBatchRequests.size();
        (void)count;
    }

    if (nullptr != req && InferenceEngine::OK == req->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY)) {
        auto output = req->GetBlob(outputDataBlobName);

        float* data_ptr = output->buffer();
        InferenceEngine::SizeVector svec = output->dims();
        size_t total = 1;
        for (size_t j = 0; j < svec.size(); j++) {
            total *= svec[j];
        }
        assert(svec[svec.size()-1] == vframes.size());

        for (size_t b = 0 ; b < vframes.size(); b++) {
            vframes[b]->detections.clear();
        }

        for (size_t i = 0; i < total; i+=7) {
            float conf = data_ptr[i + 2];
            if (conf > confidenceThreshold) {
                int idx_in_batch = static_cast<int>(data_ptr[i]);
                float x0 = std::min(std::max(0.0f, data_ptr[i + 3]), 1.0f);
                float y0 = std::min(std::max(0.0f, data_ptr[i + 4]), 1.0f);
                float x1 = std::min(std::max(0.0f, data_ptr[i + 5]), 1.0f);
                float y1 = std::min(std::max(0.0f, data_ptr[i + 6]), 1.0f);

                cv::Rect2f rect = {x0 , y0, x1-x0, y1-y0};
                vframes[idx_in_batch]->detections.push_back(Face(rect, conf, 0, 0));
            }
        }
        if (perf_timer_infer.enabled()) {
            auto end_time = std::chrono::high_resolution_clock::now();
            perf_timer_infer.addValue(end_time - start_time);
        }
    }

    if (nullptr != req) {
        std::unique_lock<std::mutex> lock(mutexPush);
        availableRequests.push(std::move(req));
        lock.unlock();
        condVarPush.notify_one();
    }

    return std::move(vframes);
}


IEGraph::~IEGraph() {
    terminate = true;
    {
        std::unique_lock<std::mutex> lock(mutexPush);
        while (!busyBatchRequests.empty()) {
            auto& req = busyBatchRequests.front().req;
            if (nullptr != req) {
                req->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
            }
            busyBatchRequests.pop();
        }
        condVarPush.notify_one();
    }
    if (getterThread.joinable()) {
        getterThread.join();
    }
}

IEGraph::Stats IEGraph::getStats() const {
    return Stats{perf_timer_preprocess.getValue(), perf_timer_infer.getValue()};
}
