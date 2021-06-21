/*
// Copyright (C) 2018-2021 Intel Corporation
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

#include "pipelines/async_pipeline.h"
#include <utils/common.hpp>
#include <utils/slog.hpp>

using namespace InferenceEngine;

AsyncPipeline::AsyncPipeline(std::unique_ptr<ModelBase>&& modelInstance, const CnnConfig& cnnConfig, InferenceEngine::Core& core) :
    model(std::move(modelInstance)) {

    execNetwork = model->loadExecutableNetwork(cnnConfig, core);

    // --------------------------- Create infer requests ------------------------------------------------
    unsigned int nireq = cnnConfig.maxAsyncRequests;
    if (nireq == 0) {
        try {
            // +1 to use it as a buffer of the pipeline
            nireq = execNetwork.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>() + 1;
        } catch (const Exception& ex) {
            throw std::runtime_error(std::string("Every device used with the demo should support "
                "OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. Failed to query the metric with error: ") + ex.what());
        }
    }
    requestsPool.reset(new RequestsPool(execNetwork, nireq));

    // --------------------------- Call onLoadCompleted to complete initialization of model -------------
    model->onLoadCompleted(requestsPool->getInferRequestsList());
}

AsyncPipeline::~AsyncPipeline() {
    waitForTotalCompletion();
}

void AsyncPipeline::waitForData(bool shouldKeepOrder) {
    std::unique_lock<std::mutex> lock(mtx);

    condVar.wait(lock, [&] {return callbackException != nullptr ||
        requestsPool->isIdleRequestAvailable() ||
        (shouldKeepOrder ?
            completedInferenceResults.find(outputFrameId) != completedInferenceResults.end() :
            !completedInferenceResults.empty());
    });

    if (callbackException)
        std::rethrow_exception(callbackException);
}

int64_t AsyncPipeline::submitData(const InputData& inputData, const std::shared_ptr<MetaData>& metaData) {
    auto frameID = inputFrameId;

    auto request = requestsPool->getIdleRequest();
    if (!request)
        return -1;

    auto startTime = std::chrono::steady_clock::now();
    auto internalModelData = model->preprocess(inputData, request);
    preprocessMetrics.update(startTime);

    request->SetCompletionCallback(
        [this, frameID, request, internalModelData, metaData, startTime] {
            request->SetCompletionCallback([]{});

            {
                std::lock_guard<std::mutex> lock(mtx);
                this->inferenceMetrics.update(startTime);
                try {
                    InferenceResult result;

                    result.frameId = frameID;
                    result.metaData = std::move(metaData);
                    result.internalModelData = std::move(internalModelData);

                    for (const auto& outName : model->getOutputsNames())
                    {
                        auto blobPtr = request->GetBlob(outName);

                        if(Precision::I32 == blobPtr->getTensorDesc().getPrecision())
                            result.outputsData.emplace(outName, std::make_shared<TBlob<int>>(*as<TBlob<int>>(blobPtr)));
                        else
                            result.outputsData.emplace(outName, std::make_shared<TBlob<float>>(*as<TBlob<float>>(blobPtr)));
                    }

                    completedInferenceResults.emplace(frameID, result);
                    this->requestsPool->setRequestIdle(request);
                }
                catch (...) {
                    if (!this->callbackException) {
                        this->callbackException = std::current_exception();
                    }
                }
            }

            condVar.notify_one();
    });

    inputFrameId++;
    if (inputFrameId < 0)
        inputFrameId = 0;

    request->StartAsync();

    return frameID;
}

std::unique_ptr<ResultBase> AsyncPipeline::getResult(bool shouldKeepOrder) {
    auto infResult = AsyncPipeline::getInferenceResult(shouldKeepOrder);
    if (infResult.IsEmpty()) {
        return std::unique_ptr<ResultBase>();
    }
    auto startTime = std::chrono::steady_clock::now();
    auto result = model->postprocess(infResult);
    postprocessMetrics.update(startTime);

    *result = static_cast<ResultBase&>(infResult);
    return result;
}

InferenceResult AsyncPipeline::getInferenceResult(bool shouldKeepOrder) {
    InferenceResult retVal;
    {
        std::lock_guard<std::mutex> lock(mtx);

        const auto& it = shouldKeepOrder ?
            completedInferenceResults.find(outputFrameId) :
            completedInferenceResults.begin();

        if (it != completedInferenceResults.end()) {
            retVal = std::move(it->second);
            completedInferenceResults.erase(it);
        }
    }

    if(!retVal.IsEmpty()) {
        outputFrameId = retVal.frameId;
        outputFrameId++;
        if (outputFrameId < 0)
            outputFrameId = 0;
    }

    return retVal;
}
