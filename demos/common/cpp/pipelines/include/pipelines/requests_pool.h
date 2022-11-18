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

#pragma once

#include <stddef.h>

#include <mutex>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>

/// This is class storing requests pool for asynchronous pipeline
///
class RequestsPool {
public:
    RequestsPool(ov::CompiledModel& compiledModel, unsigned int size);
    ~RequestsPool();

    /// Returns idle request from the pool. Returned request is automatically marked as In Use (this status will be
    /// reset after request processing completion) This function is thread safe as long as request is used only until
    /// setRequestIdle call
    /// @returns pointer to request with idle state or nullptr if all requests are in use.
    ov::InferRequest getIdleRequest();

    /// Sets particular request to Idle state
    /// This function is thread safe as long as request provided is not used after call to this function
    /// @param request - request to be returned to idle state
    void setRequestIdle(const ov::InferRequest& request);

    /// Returns number of requests in use. This function is thread safe.
    /// @returns number of requests in use
    size_t getInUseRequestsCount();

    /// Returns number of requests in use. This function is thread safe.
    /// @returns number of requests in use
    bool isIdleRequestAvailable();

    /// Waits for completion of every non-idle requests in pool.
    /// getIdleRequest should not be called together with this function or after it to avoid race condition or invalid
    /// state
    /// @returns number of requests in use
    void waitForTotalCompletion();

    /// Returns list of all infer requests in the pool.
    /// @returns list of all infer requests in the pool.
    std::vector<ov::InferRequest> getInferRequestsList();

private:
    std::vector<std::pair<ov::InferRequest, bool>> requests;
    size_t numRequestsInUse;
    std::mutex mtx;
};
