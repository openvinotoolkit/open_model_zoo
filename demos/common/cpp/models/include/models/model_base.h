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
#include <utils/args_helper.hpp>
#include <utils/ocv_common.hpp>
#include <utils/config_factory.h>
#include "input_data.h"
#include "results.h"

class ModelBase {
public:
    ModelBase(const std::string& modelFileName, const std::string& layout = "")
        : modelFileName(modelFileName) {
        layouts = parseLayoutString(layout);
    }

    virtual ~ModelBase() {}

    virtual std::shared_ptr<InternalModelData> preprocess(const InputData& inputData, ov::InferRequest& request) = 0;
    virtual ov::CompiledModel compileModel(const CnnConfig& cnnConfig, ov::Core& core);
    virtual void onLoadCompleted(const std::vector<std::shared_ptr<ov::InferRequest>>& requests) {}
    virtual std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) = 0;

    const std::vector<std::string>& getOutputsNames() const { return outputsNames; }
    const std::vector<std::string>& getInputsNames() const { return inputsNames; }

    std::string getModelFileName() { return modelFileName; }

    void setInputsPreprocessing(bool reverseInputChannels, const std::string &meanValues, const std::string &scaleValues) {
        this->inputTransform = InputTransform(reverseInputChannels, meanValues, scaleValues);
    }

protected:
    virtual void prepareInputsOutputs(std::shared_ptr<ov::Model>& model) = 0;

    std::shared_ptr<ov::Model> prepareModel(ov::Core& core);

    InputTransform inputTransform = InputTransform();
    std::vector<std::string> inputsNames;
    std::vector<std::string> outputsNames;
    ov::CompiledModel compiledModel;
    std::string modelFileName;
    CnnConfig cnnConfig = {};
    std::map<std::string, ov::Layout> layouts;
};
