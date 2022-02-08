/*
// Copyright (C) 2020-2022 Intel Corporation
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

#include <vector>
#include <openvino/openvino.hpp>
#include "pipelines/requests_pool.h"

RequestsPool::RequestsPool(ov::runtime::CompiledModel& compiledModel, unsigned int size) :
    numRequestsInUse(0) {
    for (unsigned int infReqId = 0; infReqId < size; ++infReqId) {
        requests.emplace(std::make_shared<ov::runtime::InferRequest>(compiledModel.create_infer_request()), false);
    }
}

RequestsPool::~RequestsPool() {
    // Setting empty callback to free resources allocated for previously assigned lambdas
    for (auto& pair : requests) {
        pair.first->set_callback([](const std::exception_ptr& e) {});
    }
}

RequestsPool::InferRequestPtr RequestsPool::getIdleRequest() {
    std::lock_guard<std::mutex> lock(mtx);

    const auto& it = std::find_if(requests.begin(), requests.end(), [](std::pair<const InferRequestPtr, bool>& x) {return !x.second; });
    if (it == requests.end()) {
        return std::make_shared<ov::runtime::InferRequest>(ov::runtime::InferRequest());
    }
    else {
        it->second = true;
        numRequestsInUse++;
        return it->first;
    }
}

void RequestsPool::setRequestIdle(const InferRequestPtr& request) {
    std::lock_guard<std::mutex> lock(mtx);
    this->requests.at(request) = false;
    numRequestsInUse--;
}

size_t RequestsPool::getInUseRequestsCount() {
    std::lock_guard<std::mutex> lock(mtx);
    return numRequestsInUse;
}

bool RequestsPool::isIdleRequestAvailable() {
    std::lock_guard<std::mutex> lock(mtx);
    return numRequestsInUse < requests.size();
}

void RequestsPool::waitForTotalCompletion() {
    // Do not synchronize here to avoid deadlock (despite synchronization in other functions)
    // Request status will be changed to idle in callback,
    // upon completion of request we're waiting for. Synchronization is applied there
    for (auto& pair : requests) {
        if (pair.second) {
            pair.first->wait();
        }
    }
}

std::vector<RequestsPool::InferRequestPtr> RequestsPool::getInferRequestsList() {
    std::lock_guard<std::mutex> lock(mtx);
    std::vector<InferRequestPtr> retVal;
    retVal.reserve(requests.size());
    for (auto& pair : requests) {
        retVal.push_back(pair.first);
    }

    return retVal;
}
