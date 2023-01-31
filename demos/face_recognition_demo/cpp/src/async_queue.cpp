// Copyright (C) 2023 KNS Group LLC (YADRO)
// SPDX-License-Identifier: Apache-2.0
//

#include "async_queue.hpp"
#include <openvino/openvino.hpp>
#include "utils/ocv_common.hpp"
#include <vector>

AsyncInferQueue::AsyncInferQueue(ov::CompiledModel& compiled_model, size_t size) {
    requests.resize(size);
    for (size_t request_id = 0; request_id < size; ++request_id) {
        requests[request_id] = compiled_model.create_infer_request();
        idsOfFreeRequests.push(request_id);
    }

    for (const auto& output: compiled_model.outputs()) {
        outputNames.push_back(output.get_any_name());
    }

    this->setCallback();
}

void AsyncInferQueue::setCallback() {
    for (size_t requestId = 0; requestId < requests.size(); ++requestId) {
        requests[requestId].set_callback([this, requestId /* ... */](std::exception_ptr exception_ptr) {
            {
                // acquire the mutex to access m_idle_handles
                std::lock_guard<std::mutex> lock(mutex);

                for (const auto& outName : outputNames) {
                    auto tensor = requests[requestId].get_tensor(outName);
                    results[requestId][outName] = tensor;
                }
                // Add idle handle to queue
                idsOfFreeRequests.push(requestId);
            }
            // Notify locks in getIdleRequestId()
            cv.notify_one();
            try {
                if (exception_ptr) {
                    std::rethrow_exception(exception_ptr);
                }
            } catch (const std::exception& e) {
                throw ov::Exception(e.what());
            }
        });
    }
}

AsyncInferQueue::~AsyncInferQueue() {
    waitAll();
}

size_t AsyncInferQueue::getIdleRequestId() {
    std::unique_lock<std::mutex> lock(mutex);

    cv.wait(lock, [this] {
        return !(idsOfFreeRequests.empty());
    });
    size_t idleHandle = idsOfFreeRequests.front();
    // wait for request to make sure it returned from callback
    requests[idleHandle].wait();

    return idleHandle;
}

void AsyncInferQueue::waitAll() {
    for (auto&& request : requests) {
        request.wait();
    }
}

void AsyncInferQueue::submitData(std::unordered_map<std::string,  cv::Mat> inputs, size_t input_id) {
    size_t id = getIdleRequestId();

    {
        std::lock_guard<std::mutex> lock(mutex);
        idsOfFreeRequests.pop();
    }
    requests[id].set_callback([this, id, input_id /* ... */](std::exception_ptr exception_ptr) {
        {
            // acquire the mutex to access m_idle_handles
            std::lock_guard<std::mutex> lock(mutex);
            for (const auto& outName : outputNames) {
                auto tensor = requests[id].get_tensor(outName);
                results[input_id][outName] = tensor;
            }
            // Add idle handle to queue
            idsOfFreeRequests.push(id);
        }
        // Notify locks in getIdleRequestId()
        cv.notify_one();
        try {
            if (exception_ptr) {
                std::rethrow_exception(exception_ptr);
            }
        } catch (const std::exception& e) {
            throw ov::Exception(e.what());
        }
    });
    for (const auto& input: inputs) {
        ov::Tensor inputTensor = requests[id].get_tensor(input.first);
        resize2tensor(input.second, inputTensor);
        requests[id].set_tensor(input.first, inputTensor);
    }
    requests[id].start_async();
}

std::unordered_map<int64_t, std::map<std::string, ov::Tensor>> AsyncInferQueue::getResults() {
    waitAll();
    return results;
}
