// Copyright (C) 2023 KNS Group LLC (YADRO)
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

class AsyncInferQueue {
public:
    AsyncInferQueue(ov::CompiledModel& compiled_model, size_t size);
    ~AsyncInferQueue();

    void submitData(std::unordered_map<std::string, cv::Mat> inputs, size_t input_id);
    void waitAll();
    std::unordered_map<int64_t, std::map<std::string, ov::Tensor>> getResults();

private:
    std::vector<ov::InferRequest> requests;
    std::queue<size_t> idsOfFreeRequests;
    std::unordered_map<int64_t, std::map<std::string, ov::Tensor>> results;
    std::vector<std::string> outputNames;
    void setCallback();
    size_t getIdleRequestId();
    std::mutex mutex;
    std::condition_variable cv;
};
