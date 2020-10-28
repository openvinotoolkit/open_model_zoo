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
#include "input_data.h"
#include "results.h"
#include "requests_pool.h"

class ModelBase
{
public:
    ModelBase(std::string modelFileName) { this->modelFileName = modelFileName; }
    virtual ~ModelBase() {}

    virtual void prepareInputsOutputs(InferenceEngine::CNNNetwork & cnnNetwork) = 0;
    virtual void preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request, MetaData*& metaData) = 0;
    virtual std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) = 0;
    virtual void onLoadCompleted(InferenceEngine::ExecutableNetwork* execNetwork, RequestsPool* requestsPool) {
        this->execNetwork = execNetwork; }
    virtual cv::Mat renderData(ResultBase* result) { return cv::Mat(); }
    const std::vector<std::string>& getOutputsNames() const { return outputsNames; }
    const std::vector<std::string>& getInputsNames() const { return inputsNames; }

    std::string getModelFileName() {return modelFileName; }

protected:
    std::vector<std::string> inputsNames;
    std::vector<std::string> outputsNames;
    InferenceEngine::ExecutableNetwork* execNetwork;
    std::string modelFileName;
};

