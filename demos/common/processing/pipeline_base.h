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
#include "requests_pool.h"

/// This is base class for asynchronous pipeline
/// Derived classes should add functions for data submission and output processing
class PipelineBase
{
public:
    struct ResultBase {
        int64_t frameId=-1;
        cv::Mat extraData;

        bool IsEmpty() { return frameId < 0; }
    };

    struct InferenceResult: public ResultBase {
        std::map<std::string,InferenceEngine::MemoryBlob::Ptr> outputsData;
        std::chrono::steady_clock::time_point startTime;

        /// Returns pointer to first output blob
        /// This function is a useful addition to direct access to outputs list as many models have only one output
        /// @returns pointer to first output blob
        InferenceEngine::MemoryBlob::Ptr getFirstOutputBlob() {
            if (outputsData.empty())
                throw std::out_of_range("Outputs map is empty.");
            return outputsData.begin()->second;
        }

        /// Returns true if object contains no valid data
        /// @returns true if object contains no valid data
        bool IsEmpty() { return outputsData.empty(); }
    };

    struct PerformanceInfo
    {
        int64_t framesCount = 0;
        std::chrono::steady_clock::duration latencySum = std::chrono::steady_clock::duration::zero();
        std::chrono::steady_clock::time_point startTime;
        uint32_t numRequestsInUse;
        double FPS=0;
    };

public:
    PipelineBase();
    virtual ~PipelineBase();

    /// Loads model and performs required initialization
    /// @param model_name name of model to load
    /// @param cnnConfig - fine tuning configuration for CNN model
    /// @param engine - pointer to InferenceEngine::Core instance to use.
    /// If it is omitted, new instance of InferenceEngine::Core will be created inside.
    void init(const std::string& model_name, const CnnConfig& cnnConfig, InferenceEngine::Core* engine = nullptr);

    /// Waits until output data becomes available
    ///
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
    void waitForTotalCompletion() { requestsPool->waitForTotalCompletion(); }

    /// Submit request to network
    /// @param image - image to submit for processing
    /// @returns -1 if image cannot be scheduled for processing (there's no any free InferRequest available).
    /// Otherwise reqturns unique sequential frame ID for this particular request. Same frame ID will be written in the responce structure.
    virtual int64_t submitImage(cv::Mat img);

    /// Gets available data from the queue and renders it to output frame
    /// This function should be overriden in inherited classes to provide default rendering of processed data
    /// @returns rendered frame, its size corresponds to the size of network output
    virtual cv::Mat obtainAndRenderData() { return cv::Mat(); }

protected:
    /// This function is called during intialization before loading model to device
    /// Inherited classes may override this function to prepare input/output blobs (get names, set precision, etc...)
    /// The value of outputName member variable is also may to be set here (however, it can be done in any other place).
    /// @param cnnNetwork - CNNNetwork object already loaded during initialization
    virtual void PrepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {}

    /// Submit request to network
    /// @param request - request to be submitted (caller function should obtain it using getIdleRequest)
    /// @param extraData - additional source data. This is optional transparent data not used in inference process directly.
    /// It is passed to inference result directly and can be used in postprocessing.
    /// @returns unique sequential frame ID for this particular request. Same frame ID will be written in the responce structure.
    virtual int64_t submitRequest(InferenceEngine::InferRequest::Ptr request,cv::Mat extraData =cv::Mat());

    /// Returns processed result, if available
    /// @returns InferenceResult with processed information or empty InferenceResult (with negative frameID) if there's no any results yet.
    virtual InferenceResult getInferenceResult();

protected:
    std::unique_ptr<RequestsPool> requestsPool;
    std::unordered_map<int64_t, InferenceResult> completedInferenceResults;

    InferenceEngine::ExecutableNetwork execNetwork;

    PerformanceInfo perfInfo;

    std::mutex mtx;
    std::condition_variable condVar;

    int64_t inputFrameId;
    int64_t outputFrameId;

    std::string imageInputName;
    std::vector<std::string> outputsNames;
    bool useInputAutoResize;

    std::exception_ptr callbackException = nullptr;

    /// Callback firing after request is processed by CNN
    /// NOTE: this callback is executed in separate inference engine's thread
    /// So it should not block execution for long time and should use data synchroniztion
    virtual void onProcessingCompleted(InferenceEngine::InferRequest::Ptr request) {}
};

