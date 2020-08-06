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
#include <string>
#include <deque>
#include <ngraph/ngraph.hpp>
#include <samples/ocv_common.hpp>
#include <map>
#include <condition_variable>
#include "config_factory.h"

/// This is base class for asynchronous pipeline
/// Derived classes should add functions for data submission and output processing
class PipelineBase
{
public:
    struct RequestResult {
        int64_t frameId;
        InferenceEngine::MemoryBlob::Ptr output=nullptr;
        std::chrono::steady_clock::time_point startTime;

        bool IsEmpty() { return output == nullptr; }
    };

    struct PerformanceInfo
    {
        int64_t framesCount = 0;
        std::chrono::steady_clock::duration latencySum;
        std::chrono::steady_clock::time_point startTime;
        uint32_t numRequestsInUse;
        double FPS=0;
    };

public:
    PipelineBase();
    virtual ~PipelineBase();

    /// Loads model and performs required initialization
    /// @param model_name name of model to load
    virtual void init(const std::string& model_name, const CnnConfig& cnnConfig, InferenceEngine::Core* engine=nullptr);

    /// Waits until output data becomes available
    ///
    void waitForData();

    /// Returns performance info
    /// @returns performance information structure
    PerformanceInfo getPerformanceInfo() { std::lock_guard<std::mutex> lock(mtx); return perfInfo; }

    /// Waits for all currently submitted requests to be completed.
    ///
    void waitForCompletion();

protected:
    /// This function is called during intialization before loading model to device
    /// Inherited classes may override this function to prepare input/output blobs (get names, set precision, etc...)
    /// The value of outputName member variable is also may to be set here (however, it can be done in any other place).
    /// @param cnnNetwork - CNNNetwork object already loaded during initialization
    virtual void PrepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {}

    /// Submit request to network
    /// @param request - request to be submitted (caller function should obtain it using getIdleRequest)
    /// @returns unique sequential frame ID for this particular request. Same frame ID will be written in responce structure.
    virtual int64_t submitRequest(InferenceEngine::InferRequest::Ptr request);

    /// Returns idle request from the pool. Returned request is automatically marked as In Use (this status will be reset after request processing completion)
    /// @returns pointer to request with idle state or nullptr if all requests are in use.
    InferenceEngine::InferRequest::Ptr getIdleRequest();

    /// Returns processed result, if available
    /// @returns RequestResult with processed information or empty RequestResult (with negative frameID) if there's no any results yet.
    virtual RequestResult getResult();

protected:

    std::map<InferenceEngine::InferRequest::Ptr, std::atomic_bool> requestsPool;
    std::unordered_map<int64_t, RequestResult> completedRequestResults;

    InferenceEngine::ExecutableNetwork execNetwork;

    PerformanceInfo perfInfo;

    std::mutex mtx;
    std::condition_variable condVar;

    int64_t inputFrameId;
    int64_t outputFrameId;
    std::string outputName;

    std::exception_ptr callbackException = nullptr;

    bool isRequestsPoolEmpty(){
        return std::find_if(requestsPool.begin(), requestsPool.end(), [](std::pair<const InferenceEngine::InferRequest::Ptr,std::atomic_bool>& x) {return !x.second; })==requestsPool.end();
    }

    void setRequestIdle(const InferenceEngine::InferRequest::Ptr& request) {
        this->requestsPool.at(request) = false;
    }

    int64_t getInUseRequestsCount() {
        return std::count_if(requestsPool.begin(), requestsPool.end(), [](std::pair<const InferenceEngine::InferRequest::Ptr, std::atomic_bool>& x) {return (bool)x.second; });
    }

};

