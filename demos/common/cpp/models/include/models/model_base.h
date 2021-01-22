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

#pragma once
#include "input_data.h"
#include "results.h"

class ModelBase {
public:
    ModelBase(const std::string& modelFileName)
        : execNetwork(nullptr), modelFileName(modelFileName)
    {}

    virtual ~ModelBase() {}

    virtual void prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) = 0;
    virtual std::shared_ptr<InternalModelData> preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) = 0;
    virtual std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) = 0;
        virtual void onLoadCompleted(InferenceEngine::ExecutableNetwork* execNetwork, const std::vector<InferenceEngine::InferRequest::Ptr>& requests) {
        this->execNetwork = execNetwork; }
    const std::vector<std::string>& getOutputsNames() const { return outputsNames; }
    const std::vector<std::string>& getInputsNames() const { return inputsNames; }

    std::string getModelFileName() { return modelFileName; }

    virtual void reshape(InferenceEngine::CNNNetwork & cnnNetwork) {
        auto shapes = cnnNetwork.getInputShapes();
        for (auto& shape : shapes)
            shape.second[0] = 1;
        cnnNetwork.reshape(shapes);
    }

protected:
    std::vector<std::string> inputsNames;
    std::vector<std::string> outputsNames;
    InferenceEngine::ExecutableNetwork* execNetwork;
    std::string modelFileName;
};
