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
#include <openvino/openvino.hpp>
#include <utils/ocv_common.hpp>
#include <utils/config_factory.h>

#include "input_data.h"
#include "results.h"

class ModelBase {
public:
    ModelBase(const std::string& modelFileName)
        : modelFileName(modelFileName)
    {}

    virtual ~ModelBase() {}

    virtual std::shared_ptr<InternalModelData> preprocess(const InputData& inputData, ov::runtime::InferRequest& request) = 0;
    virtual std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) = 0;
    virtual void onLoadCompleted(const std::vector<std::shared_ptr<ov::runtime::InferRequest>>& requests) {}
    const std::vector<std::string>& getOutputsNames() const { return outputsNames; }
    const std::vector<std::string>& getInputsNames() const { return inputsNames; }

    virtual ov::runtime::CompiledModel compileModel(const CnnConfig& cnnConfig, ov::runtime::Core& core);

    std::string getModelFileName() { return modelFileName; }

    void SetInputsPreprocessing(bool reverseInputChannels, const std::string &meanValues, const std::string &scaleValues) {
        this->inputTransform = InputTransform(reverseInputChannels, meanValues, scaleValues);
    }

    void setBatchOne(std::shared_ptr<ov::Model>& model) {
        ov::set_batch(model, 1);
    }

protected:
    std::shared_ptr<ov::Model> prepareModel(ov::runtime::Core& core);
    virtual void prepareInputsOutputs(std::shared_ptr<ov::Model>& model) = 0;

    InputTransform inputTransform = InputTransform();
    std::vector<std::string> inputsNames;
    std::vector<std::string> outputsNames;
    ov::runtime::CompiledModel compiledModel;
    std::string modelFileName;
    CnnConfig cnnConfig = {};
};
