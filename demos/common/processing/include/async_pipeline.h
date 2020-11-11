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
#include <map>
#include <condition_variable>
#include "config_factory.h"
#include "requests_pool.h"
#include "results.h"
#include "model_base.h"
#include "samples/performance_metrics.hpp"

/// This is base class for asynchronous pipeline
/// Derived classes should add functions for data submission and output processing
class AsyncPipeline
{
public:
    /// Loads model and performs required initialization
    /// @param modelInstance pointer to model object. Object it points to should not be destroyed manually after passing pointer to this function.
    /// @param cnnConfig - fine tuning configuration for CNN model
    /// @param engine - reference to InferenceEngine::Core instance to use.
    /// If it is omitted, new instance of InferenceEngine::Core will be created inside.
    AsyncPipeline(std::unique_ptr<ModelBase>&& modelInstance, const CnnConfig& cnnConfig, InferenceEngine::Core& engine);
    virtual ~AsyncPipeline();

    /// Waits until either output data becomes available or pipeline allows to submit more input data.
    /// Function will treat results as ready only if next sequential result (frame) is ready.
    void waitForData();

    /// Returns true if there's available infer requests in the pool
    /// and next frame can be submitted for processing.
    /// @returns true if there's available infer requests in the pool
    /// and next frame can be submitted for processing, false otherwise.
    bool isReadyToProcess() { return requestsPool->isIdleRequestAvailable(); }

    /// Returns performance metrics
    /// @returns performance metrics structure
    const PerformanceMetrics getMetrics() const { return perfMetrics; }

    /// Waits for all currently submitted requests to be completed.
    ///
    void waitForTotalCompletion() { if (requestsPool) requestsPool->waitForTotalCompletion(); }

    /// Submits data to the network for inference
    /// @param inputData - input data to be submitted
    /// @returns -1 if image cannot be scheduled for processing (there's no free InferRequest available).
    /// Otherwise returns unique sequential frame ID for this particular request. Same frame ID will be written in the response structure.
    virtual int64_t submitData(const InputData& inputData);

    /// Gets available data from the queue
    /// Function will treat results as ready only if next sequential result (frame) is ready.
    virtual std::unique_ptr<ResultBase> getResult();

protected:
    /// Returns processed result, if available
    /// Function will treat results as ready only if next sequential result (frame) is ready.
    /// @returns InferenceResult with processed information or empty InferenceResult (with negative frameID) if there's no any results yet.
    virtual InferenceResult getInferenceResult();

protected:
    std::unique_ptr<RequestsPool> requestsPool;
    std::unordered_map<int64_t, InferenceResult> completedInferenceResults;

    InferenceEngine::ExecutableNetwork execNetwork;

    PerformanceMetrics perfMetrics;

    std::mutex mtx;
    std::condition_variable condVar;

    int64_t inputFrameId=0;
    int64_t outputFrameId=0;

    std::exception_ptr callbackException = nullptr;

    std::unique_ptr<ModelBase> model;
};
