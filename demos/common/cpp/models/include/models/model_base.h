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
#include <utils/args_helper.hpp>
#include "utils/config_factory.h"

class ModelBase {
public:
    ModelBase(const std::string& modelFileName)
        : modelFileName(modelFileName), isNetworkCompiled(fileExt(modelFileName) == "blob")
    {}

    virtual ~ModelBase() {}

    virtual std::shared_ptr<InternalModelData> preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) = 0;
    virtual std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) = 0;
    virtual void onLoadCompleted(const std::vector<InferenceEngine::InferRequest::Ptr>& requests) {}
    const std::vector<std::string>& getOutputsNames() const { return outputsNames; }
    const std::vector<std::string>& getinputsNames() const { return inputsNames; }

    virtual InferenceEngine::ExecutableNetwork loadExecutableNetwork(const CnnConfig& cnnConfig, InferenceEngine::Core& core);

    std::string getModelFileName() { return modelFileName; }

    void setBatchOne(InferenceEngine::CNNNetwork& cnnNetwork) {
        auto shapes = cnnNetwork.getInputShapes();
        for (auto& shape : shapes)
            shape.second[0] = 1;
        cnnNetwork.reshape(shapes);
    }

    struct IOPattern {
        const std::map<size_t, std::map<std::string, InferenceEngine::TensorDesc>> possibleIO;

        IOPattern(std::map<size_t, std::map<std::string, InferenceEngine::TensorDesc>>&& inputsOrOutputs)
            : possibleIO(std::move(inputsOrOutputs)) {}

    };

protected:
    InferenceEngine::CNNNetwork prepareNetwork(InferenceEngine::Core& core);

    template<class InputsDataMap, class OutputsDataMap>
    void findIONames(const InputsDataMap& inputsInfo, const OutputsDataMap& outputInfo);

    template<class InputsDataMap, class OutputsDataMap>
    void checkInputsOutputs(const std::pair<std::string, std::vector<ModelBase::IOPattern>>& modelIOPattern,
        const InputsDataMap& inputsInfo, const OutputsDataMap& outputsInfo);

    virtual void prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) = 0;
    virtual void checkCompiledNetworkInputsOutputs() = 0;
    std::vector<std::string> inputsNames;
    std::vector<std::string> outputsNames;
    std::vector<std::string> inputsNames_;
    std::vector<std::string> outputsNames_;
    InferenceEngine::ExecutableNetwork execNetwork;
    std::string modelFileName;
    CnnConfig cnnConfig = {};
    bool isNetworkCompiled;
};

template<class InputsDataMap, class OutputsDataMap>
void ModelBase::findIONames(const InputsDataMap& inputsInfo, const OutputsDataMap& outputInfo) {
    //--------------------------- Find input names --------------------------- 
    std::cout << "inputs" << std::endl;
    for (const auto& input : inputsInfo) {
        std::cout << input.first << std::endl;
        inputsNames_.push_back(input.first);
    }

    //--------------------------- Find outputs names --------------------------- 
    std::cout << "outputs" << std::endl;

    for (const auto& output : outputInfo) {
        std::cout << output.first << std::endl;
        outputsNames_.push_back(output.first);
    }
}

template<class DataMap>
void check(const std::string& modelName, const std::string& type, const ModelBase::IOPattern& pattern, const DataMap& info, std::vector<std::string>& names) {
    // --------------------------- Check number of I/O--------------------------- 
    if (pattern.possibleIO.find(info.size()) == pattern.possibleIO.end()) {
        std::ostringstream ossNums;
        // Convert set to string
        for (const auto& n : pattern.possibleIO) {
            ossNums << n.first << ", ";
        }

        //std::copy(pattern.possibleNum.begin(), pattern.possibleNum.end(),
        //    std::ostream_iterator<int>(ossNum, ","));

        throw std::logic_error("Number of " + type + "s in " + modelName + " network should be one of the following : " + ossNums.str() +
            " - but " + std::to_string(info.size()) + " given");
    }

    bool layerIsFound = false;
    auto& layersPatterns = pattern.possibleIO.find(info.size())->second;
    for (auto& layerName : names) {
        const auto& layerPatternDesc = layersPatterns.find(layerName);
        if (layerPatternDesc != layersPatterns.end()) {
            const auto& layerToCheck = info.find(layerName)->second;
            const auto& layerToCheckDesc = layerToCheck->getTensorDesc();
            const auto& layerToCheckDims = layerToCheckDesc.getDims();
            const auto& layerPatternDims = layerPatternDesc->second.getDims();

            if (layerToCheckDims.size() != layerPatternDims.size()) {
                throw std::logic_error("Unsupported " + std::to_string(layerToCheckDims.size()) + "D " + type + " layer '" + layerName + "'. " +
                    " In " + modelName + " network number of dimensions for '" + layerName + "' layer should be : " + std::to_string(layerPatternDims.size()));
            }

            for (size_t i = 0; i < layerToCheckDims.size(); ++i) {
                if (layerPatternDims[i] != layerToCheckDims[i] && layerPatternDims[i] != 0) {
                    std::ostringstream ossDims1, ossDims2;
                    for (size_t i = 0; i < layerToCheckDims.size(); ++i) {
                        ossDims1 << layerToCheckDims[i] << ",";
                        if (layerPatternDims[i] == 0) {
                            ossDims2 << "X,";
                        }
                        else {
                            ossDims2 << layerPatternDims[i] << ",";
                        }
                    }
                    throw std::logic_error("Unsupported " + type + " layer '" + layerName + "' with dimension [" + ossDims1.str() + "]. " +
                        " In " + modelName + " network demensions for '" + layerName + "' layer should be : [" + ossDims2.str() + "]");
                }
            }

            // --------------------------- Check precision---------------------------
            if (layerPatternDesc->second.getPrecision() != layerToCheckDesc.getPrecision()) {

                throw std::logic_error("In " + modelName + " network " +
                    type + " layer '" + layerName + "' precision should be: " + layerPatternDesc->second.getPrecision().name() + " - but " + layerToCheckDesc.getPrecision().name() + " given");
            }

            // --------------------------- Check layout--------------------------- 
            if (layerPatternDesc->second.getLayout() != layerToCheckDesc.getLayout()) {
                std::ostringstream oss;
                oss << "In " << modelName << " network " << type << " layer '" << layerName << "' layout should be "
                    << layerPatternDesc->second.getLayout() << " - but " << layerToCheckDesc.getLayout() << " given";

                throw std::logic_error(oss.str());
            }
            layerIsFound = true;
        }
    }

    if (!layerIsFound) {
        throw std::logic_error(modelName + " network has wrong " + type + " layers' names");
    }
}

template<class InputsDataMap, class OutputsDataMap>
void  ModelBase::checkInputsOutputs(const std::pair<std::string, std::vector<ModelBase::IOPattern>>& modelIOPattern,
    const InputsDataMap& inputsInfo, const OutputsDataMap& outputsInfo) {
    //--------------------------- Check inputs blobs ------------------------------------------------------
    slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
    check(modelIOPattern.first, "input", modelIOPattern.second[0], inputsInfo, inputsNames_);

    // --------------------------- Check output blobs -----------------------------------------------------
    slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
    check(modelIOPattern.first, "output", modelIOPattern.second[1], outputsInfo, outputsNames_);
}
