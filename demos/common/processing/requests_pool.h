/*
// Copyright (C) 2018-2020 Intel Corporation
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

#pragma once

#include <map>
#include <atomic>
#include <samples/ocv_common.hpp>

/// This is class storing requests pool for asynchronous pipeline
/// 
class RequestsPool
{
public:
    RequestsPool(){}

    RequestsPool(InferenceEngine::ExecutableNetwork& execNetwork, unsigned int size);

    /// Returns idle request from the pool. Returned request is automatically marked as In Use (this status will be reset after request processing completion)
    /// This function is thread safe as long as request is used only until setRequestIdle call
    /// @returns pointer to request with idle state or nullptr if all requests are in use.
    InferenceEngine::InferRequest::Ptr getIdleRequest();

    /// Sets particular request to Idle state
    /// This function is thread safe as long as request provided is not used after call to this function
    /// @param request - request to be returned to idle state
    void setRequestIdle(const InferenceEngine::InferRequest::Ptr& request) {
        this->requestsPool.at(request) = false;
    }

    /// Returns number of requests in use. This function is thread safe.
    /// @returns number of requests in use
    int64_t getInUseRequestsCount() {
        return std::count_if(requestsPool.begin(), requestsPool.end(), [](std::pair<const InferenceEngine::InferRequest::Ptr, std::atomic_bool>& x) {return (bool)x.second; });
    }

    /// Returns number of requests in use. This function is thread safe.
    /// @returns number of requests in use
    bool isIdleRequestAvailable() {
        for (auto it = requestsPool.begin(); it != requestsPool.end(); it++) {
            if (!it->second) {
                return true;
            }
        }
        return false;
    }

    /// Waits for completion of every non-idle requests in pool.
    /// getIdleRequest should not be called together with this function or after it to avoid race condition or invalid state
    /// @returns number of requests in use
    void waitForTotalCompletion();

    /// Returns iterator pointing to the start of the pool
    /// @returns iterator pointing to the start of the pool
    std::map<InferenceEngine::InferRequest::Ptr, std::atomic_bool>::iterator begin() { return requestsPool.begin(); }

    /// Returns iterator pointing to the end of the pool
    /// @returns iterator pointing to the end of the pool
    std::map<InferenceEngine::InferRequest::Ptr, std::atomic_bool>::iterator end() { return requestsPool.end(); }

private:
    std::map<InferenceEngine::InferRequest::Ptr, std::atomic_bool> requestsPool;
};
