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

#include "pipelines/async_pipeline.h"

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>

#include <models/model_base.h>
#include <models/results.h>
#include <utils/config_factory.h>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>

struct InputData;
struct MetaData;

AsyncPipeline::AsyncPipeline(std::unique_ptr<ModelBase>&& modelInstance, const ModelConfig& config, ov::Core& core)
    : model(std::move(modelInstance)) {
    compiledModel = model->compileModel(config, core);
    // --------------------------- Create infer requests ------------------------------------------------
    unsigned int nireq = config.maxAsyncRequests;
    if (nireq == 0) {
        try {
            // +1 to use it as a buffer of the pipeline
            nireq = compiledModel.get_property(ov::optimal_number_of_infer_requests) + 1;
        } catch (const ov::Exception& ex) {
            throw std::runtime_error(
                std::string("Every device used with the demo should support compiled model's property "
                            "'OPTIMAL_NUMBER_OF_INFER_REQUESTS'. Failed to query the property with error: ") +
                ex.what());
        }
    }
    slog::info << "\tNumber of inference requests: " << nireq << slog::endl;
    requestsPool.reset(new RequestsPool(compiledModel, nireq));
    // --------------------------- Call onLoadCompleted to complete initialization of model -------------
    model->onLoadCompleted(requestsPool->getInferRequestsList());
}

AsyncPipeline::~AsyncPipeline() {
    waitForTotalCompletion();
}

void AsyncPipeline::waitForData(bool shouldKeepOrder) {
    std::unique_lock<std::mutex> lock(mtx);

    condVar.wait(lock, [&]() {
        return callbackException != nullptr || requestsPool->isIdleRequestAvailable() ||
               (shouldKeepOrder ? completedInferenceResults.find(outputFrameId) != completedInferenceResults.end()
                                : !completedInferenceResults.empty());
    });

    if (callbackException) {
        std::rethrow_exception(callbackException);
    }
}

int64_t AsyncPipeline::submitData(const InputData& inputData, const std::shared_ptr<MetaData>& metaData) {
    auto frameID = inputFrameId;

    auto request = requestsPool->getIdleRequest();
    if (!request) {
        return -1;
    }

    auto startTime = std::chrono::steady_clock::now();
    auto internalModelData = model->preprocess(inputData, request);
    preprocessMetrics.update(startTime);

    request.set_callback(
        [this, request, frameID, internalModelData, metaData, startTime](std::exception_ptr ex) mutable {
            {
                const std::lock_guard<std::mutex> lock(mtx);
                inferenceMetrics.update(startTime);
                try {
                    if (ex) {
                        std::rethrow_exception(ex);
                    }
                    InferenceResult result;

                    result.frameId = frameID;
                    result.metaData = std::move(metaData);
                    result.internalModelData = std::move(internalModelData);

                    for (const auto& outName : model->getOutputsNames()) {
                        auto tensor = request.get_tensor(outName);
                        result.outputsData.emplace(outName, tensor);
                    }

                    completedInferenceResults.emplace(frameID, result);
                    requestsPool->setRequestIdle(request);
                } catch (...) {
                    if (!callbackException) {
                        callbackException = std::current_exception();
                    }
                }
            }
            condVar.notify_one();
        });

    inputFrameId++;
    if (inputFrameId < 0)
        inputFrameId = 0;

    request.start_async();

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
        const std::lock_guard<std::mutex> lock(mtx);

        const auto& it =
            shouldKeepOrder ? completedInferenceResults.find(outputFrameId) : completedInferenceResults.begin();

        if (it != completedInferenceResults.end()) {
            retVal = std::move(it->second);
            completedInferenceResults.erase(it);
        }
    }

    if (!retVal.IsEmpty()) {
        outputFrameId = retVal.frameId;
        outputFrameId++;
        if (outputFrameId < 0) {
            outputFrameId = 0;
        }
    }

    return retVal;
}
