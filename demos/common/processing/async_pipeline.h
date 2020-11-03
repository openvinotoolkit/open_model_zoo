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

/// This is base class for asynchronous pipeline
/// Derived classes should add functions for data submission and output processing
class PipelineBase
{
public:
    static constexpr int MOVING_AVERAGE_SAMPLES = 5;

    struct PerformanceInfo
    {
        int64_t framesCount = 0;
        std::chrono::steady_clock::duration latencySum = std::chrono::steady_clock::duration::zero();
        std::chrono::steady_clock::duration lastInferenceLatency = std::chrono::steady_clock::duration::zero();
        std::chrono::steady_clock::time_point startTime;
        double movingAverageLatencyMs;
        uint32_t numRequestsInUse;
        double movingAverageFPS;
        double FPS=0;

        double getTotalAverageLatencyMs() const {
            return ((double)std::chrono::duration_cast<std::chrono::milliseconds>(latencySum).count()) / framesCount;
        }

        double getLastInferenceLatencyMs() const {
            return ((double)std::chrono::duration_cast<std::chrono::milliseconds>(lastInferenceLatency).count());
        }

    };

    struct PerformanceInternalCounters
    {
        long long latenciesMs[MOVING_AVERAGE_SAMPLES] = {};
        std::chrono::steady_clock::time_point retrievalTimestamps[MOVING_AVERAGE_SAMPLES] = {};
        long long movingLatenciesSumMs = 0;
        int currentIndex = 0;
    };

public:
    /// Loads model and performs required initialization
    /// @param modelInstance pointer to model object. Object it points to should not be destroyed manually after passing pointer to this function.
    /// @param cnnConfig - fine tuning configuration for CNN model
    /// @param engine - pointer to InferenceEngine::Core instance to use.
    /// If it is omitted, new instance of InferenceEngine::Core will be created inside.
    PipelineBase(std::unique_ptr<ModelBase> modelInstance, const CnnConfig& cnnConfig, InferenceEngine::Core* engine = nullptr);
    virtual ~PipelineBase();

    /// Waits until either output data becomes available or pipeline allows to submit more input data.
    /// Function will treat results as ready only if next sequential result (frame) is ready.
    void waitForData();

    /// Returns true if there's available infer requests in the pool
    /// and next frame can be submitted for processing.
    /// @returns true if there's available infer requests in the pool
    /// and next frame can be submitted for processing, false otherwise.
    bool isReadyToProcess() { return requestsPool->isIdleRequestAvailable(); }

    /// Returns performance info
    /// @returns performance information structure
    PerformanceInfo getPerformanceInfo() { std::lock_guard<std::mutex> lock(mtx); return perfInfo; }

    /// Waits for all currently submitted requests to be completed.
    ///
    void waitForTotalCompletion() { if(requestsPool.get())requestsPool->waitForTotalCompletion(); }

    /// Submit request to network
    /// @param image - image to submit for processing
    /// @returns -1 if image cannot be scheduled for processing (there's no any free InferRequest available).
    /// Otherwise returns unique sequential frame ID for this particular request. Same frame ID will be written in the response structure.
    virtual int64_t submitImage(cv::Mat img);

    /// Gets available data from the queue 
    /// Function will treat results as ready only if next sequential result (frame) is ready.
    virtual std::unique_ptr<ResultBase> getResult();

protected:
    /// Submit request to network
    /// @param request - request to be submitted (caller function should obtain it using getIdleRequest)
    /// @param metaData - additional source data. This is optional transparent data not used in inference process directly.
    /// It is passed to inference result directly and can be used in postprocessing.
    /// @returns unique sequential frame ID for this particular request. Same frame ID will be written in the responce structure.
    virtual int64_t submitRequest(const InferenceEngine::InferRequest::Ptr& request,const std::shared_ptr<MetaData>& metaData);

    /// Returns processed result, if available
    /// Function will treat results as ready only if next sequential result (frame) is ready.
    /// @returns InferenceResult with processed information or empty InferenceResult (with negative frameID) if there's no any results yet.
    virtual InferenceResult getInferenceResult();

protected:
    std::unique_ptr<RequestsPool> requestsPool;
    std::unordered_map<int64_t, InferenceResult> completedInferenceResults;

    InferenceEngine::ExecutableNetwork execNetwork;

    PerformanceInfo perfInfo;
    PerformanceInternalCounters perfInternals;

    std::mutex mtx;
    std::condition_variable condVar;

    int64_t inputFrameId=0;
    int64_t outputFrameId=0;

    std::exception_ptr callbackException = nullptr;

    /// Callback firing after request is processed by CNN
    /// NOTE: this callback is executed in separate inference engine's thread
    /// So it should not block execution for long time and should use data synchroniztion
    virtual void onProcessingCompleted(InferenceEngine::InferRequest::Ptr request) {}

    std::unique_ptr<ModelBase> model;
};
