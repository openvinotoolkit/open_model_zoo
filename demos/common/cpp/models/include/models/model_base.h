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
        const std::set<int> possibleNums;
        using options = std::tuple<std::vector<std::string>, std::vector<InferenceEngine::Precision>,
            std::vector<InferenceEngine::SizeVector>, std::vector<InferenceEngine::Layout>>;
        const options layerOptions;
        const std::map<std::string, InferenceEngine::TensorDesc> possibleLayers;
        const std::set<std::string> skipLayers;
        void generatePossibleLayers() {

        }

        IOPattern(std::set<int>&& num, std::map<std::string, InferenceEngine::TensorDesc>&& layers, const std::set<std::string>&& skip = {})
            : possibleNums(std::move(num)), skipLayers(std::move(skip)), possibleLayers(std::move(layers)) {
            generatePossibleLayers();
        }

    };

    struct IOPattern_ {
        InferenceEngine::SizeVector dims;
        InferenceEngine::Precision precision;
        std::set<InferenceEngine::Layout> layout;
    };

    std::multimap<size_t, std::map<std::string, IOPattern_>> possibleIO;
        
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
    std::cout << "iputs" << std::endl;
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
void check(const std::string& modelName, const std::string& type, const ModelBase::IOPattern& pattern, const DataMap& info) {
    // --------------------------- Check number of I/O--------------------------- 
    if (pattern.possibleNums.find(info.size()) == pattern.possibleNums.end()) {
        std::ostringstream ossNums;
        // Convert set to string
        for (const auto& n : pattern.possibleNums) {
            ossNums<< n << ", ";
        }

        //std::copy(pattern.possibleNum.begin(), pattern.possibleNum.end(),
        //    std::ostream_iterator<int>(ossNum, ","));

        throw std::logic_error("Number of " + type + "s in " + modelName + " network should be one of the following : " + ossNums.str() +
            " - but " + std::to_string(info.size()) + " given");
    }

    size_t i = 0;
    for (const auto& layerToCheck : info) {
        if (pattern.skipLayers.find(layerToCheck.first) == pattern.skipLayers.end()) {
            const InferenceEngine::TensorDesc& layerToCheckDesc = layerToCheck.second->getTensorDesc();
                auto layerToCheckDims = layerToCheckDesc.getDims();

                // --------------------------- Check dimensions--------------------------- 
                auto& layerDesc = pattern.possibleLayers.find(layerToCheck.first);
                if (layerDesc == pattern.possibleLayers.end()) {
                    std::ostringstream ossNames;
                        for (const auto& s : pattern.possibleLayers) {
                            ossNames << s.first << ", ";
                        }

                    throw std::logic_error("In " + modelName + " network wrong " + type + " layer name: " + layerToCheck.first + ". Name should be one of the next: " + ossNames.str());
                }

            auto& layerDims = layerDesc->second.getDims();
            if (layerToCheckDims.size() != layerDims.size()) {
                throw std::logic_error("Unsupported " + std::to_string(layerToCheckDims.size()) + "D " + type + " layer '" + layerToCheck.first + "'. " +
                    " In " + modelName + " network number of dimensions for '" + layerToCheck.first + "' layer should be : " + std::to_string(layerDims.size()));
            }

            for (size_t i = 0; i < layerToCheckDims.size(); ++i) {
                if (layerDims[i] != layerToCheckDims[i] && layerDims[i] != 0) {
                    std::ostringstream ossDims1, ossDims2;
                    for (size_t i = 0; i < layerDims.size(); ++i) {
                        ossDims1 << layerToCheckDims[i] << ",";
                        if (layerDims[i] == 0) {
                            ossDims2 << "X,";
                        }
                        else {
                            ossDims2 << layerDims[i] << ",";
                        }
                    }
                    throw std::logic_error("Unsupported " + type + " layer '" + layerToCheck.first + "' with dimension [" + ossDims1.str() + "]. " +
                        " In " + modelName + " network demensions for '" + layerToCheck.first + "' layer should be : [" + ossDims2.str() + "]");
                }
            }

            // --------------------------- Check precision---------------------------
            if (layerDesc->second.getPrecision() != layerToCheckDesc.getPrecision()) {
                // std::ostringstream ossPrecisions;

                 //for (const auto& p : pattern.possiblePrecisions) {
                 //    ossPrecisions << p.name() << ", ";
                 //}

                throw std::logic_error("In " + modelName + " network " +
                    type + " layer '" + layerToCheck.first + "' precision should be: " + layerDesc->second.getPrecision().name() + " - but " + layerToCheckDesc.getPrecision().name() + " given");
            }

            // --------------------------- Check layout--------------------------- 
            if (layerDesc->second.getLayout() != layerToCheckDesc.getLayout()) {
                std::ostringstream oss;
                oss << "In " << modelName << " network " << type << " layer '" << layerToCheck.first << "' layout should be "
                    << layerDesc->second.getLayout() << " - but " << layerToCheck.second->getLayout() << " given";

                throw std::logic_error(oss.str());
            }
            ++i;
        }
    }
}

template<class InputsDataMap, class OutputsDataMap>
void  ModelBase::checkInputsOutputs(const std::pair<std::string, std::vector<ModelBase::IOPattern>>& modelIOPattern,
    const InputsDataMap& inputsInfo, const OutputsDataMap& outputsInfo) {
    //--------------------------- Check inputs blobs ------------------------------------------------------
    slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
    check(modelIOPattern.first, "input", modelIOPattern.second[0], inputsInfo);

    // --------------------------- Check output blobs -----------------------------------------------------
    slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
    check(modelIOPattern.first, "output", modelIOPattern.second[1], outputsInfo);
}
