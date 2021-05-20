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
        const std::string modelName;
        const std::set<int> possibleNums;
        const std::set<int> possibleSizes;
        const std::map<std::string, InferenceEngine::SizeVector> possibleDims;
        const std::set<InferenceEngine::Precision> possiblePrecisions;
        const std::set<InferenceEngine::Layout> possibleLayouts;

        IOPattern (std::string&& modelName, std::set<int>&& num, std::set<int>&& sizes, std::map<std::string, InferenceEngine::SizeVector>&& dims,
            std::set<InferenceEngine::Precision>&& precisions, std::set<InferenceEngine::Layout>&& layouts)
            : modelName(std::move(modelName)), possibleNums(std::move(num)), possibleSizes(std::move(sizes)), possibleDims(std::move(dims)),
            possiblePrecisions(std::move(precisions)), possibleLayouts(std::move(layouts)) {}
    };
        
protected:
    InferenceEngine::CNNNetwork prepareNetwork(InferenceEngine::Core& core);

    template<class InputsDataMap, class OutputsDataMap>
    void findIONames(const InputsDataMap& inputsInfo, const OutputsDataMap& outputInfo);

    template<class InputsDataMap, class OutputsDataMap>
    void checkInputsOutputs(const std::string& modelName, const std::vector<ModelBase::IOPattern>& modelIOPattern,
        const InputsDataMap& inputsInfo, const OutputsDataMap& outputsInfo);

    virtual void prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) = 0;
    virtual void checkCompiledNetworkInputsOutputs() = 0;
    std::vector<std::string> inputsNames;
    std::vector<std::string> outputsNames;
    InferenceEngine::ExecutableNetwork execNetwork;
    std::string modelFileName;
    CnnConfig cnnConfig = {};
    bool isNetworkCompiled;
};

template<class InputsDataMap, class OutputsDataMap>
void ModelBase::findIONames(const InputsDataMap& inputsInfo, const OutputsDataMap& outputInfo) {
    //--------------------------- Find input names --------------------------- 
    for (const auto& input : inputsInfo) {
        inputsNames.push_back(input.first);
    }

    //--------------------------- Find outputs names --------------------------- 
    for (const auto& output : outputInfo) {
        outputsNames.push_back(output.first);
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
    for (const auto& layer : info) {
        const InferenceEngine::TensorDesc& desc = layer.second->getTensorDesc();
        auto modelDims = desc.getDims();

        // --------------------------- Check dimensions--------------------------- 
        auto& dims  = pattern.possibleDims.find(layer.first);
        if (dims == pattern.possibleDims.end()) {
            std::ostringstream ossNames;

            for (const auto& s : pattern.possibleDims) {
                ossNames << s.first << ", ";
            }

            //// --- Convert set to string
            //std::copy(pattern.possiblelayerSizes.begin(), pattern.possiblelayerSizes.end(),
            //    std::ostream_iterator<int>(ossSizes, ","));
            throw std::logic_error("In " + pattern.modelName + " network wrong " + type + " layer name: " + layer.first + ". Name should be one of the next: " + ossNames.str());
        }

        if (modelDims.size() != dims->second.size()) {
            throw std::logic_error("Unsupported " + std::to_string(modelDims.size()) + "D " + type + " layer '" + layer.first + "'. " +
                " In " + pattern.modelName + " network number of dimensions for '" + layer.first + "' layer should be : " + std::to_string(dims->second.size()));
        }

        for (size_t i = 0; i < modelDims.size(); ++i) {
            if (dims->second[i] != modelDims[i] && dims->second[i] != 0) {
                std::ostringstream ossModelDims, ossDims;
                for (size_t i = 0; i < modelDims.size(); ++i) {
                    ossModelDims << modelDims[i] << ",";
                    if (dims->second[i] == 0) {
                        ossDims << "X,";
                    }
                    else {
                        ossDims << dims->second[i] << ",";
                    }
                }
                throw std::logic_error("Unsupported " + type + " layer '" + layer.first + "' with dimension [" + ossModelDims.str() + "]. " +
                    " In " + pattern.modelName + " network demensions for '" + layer.first + "' layer should be : [" + ossDims.str() + "]");
            }
        }

        // --------------------------- Check precision---------------------------
        if (pattern.possiblePrecisions.find(desc.getPrecision()) == pattern.possiblePrecisions.end()) {
            std::ostringstream ossPrecisions;

            for (const auto& p : pattern.possiblePrecisions) {
                ossPrecisions << p.name() << ", ";
            }

            //std::transform(pattern.possiblelayerPrecisions.begin(), pattern.possiblelayerPrecisions.end(),
            //    std::ostream_iterator<std::string>(osslayerPrecisions, ","),
            //    [](const InferenceEngine::Precision& p) {
            //        return p.name();
            //    });

            throw std::logic_error("In " + pattern.modelName + " network " +
                type + " layer '" + layer.first + "' precision should be one of the following : " + ossPrecisions.str() + " - but " + desc.getPrecision().name() + " given");
        }

        // --------------------------- Check layout--------------------------- 
        if (pattern.possibleLayouts.find(desc.getLayout()) == pattern.possibleLayouts.end()) {
            std::ostringstream ossLayouts;
            for (const auto& l : pattern.possibleLayouts) {
                ossLayouts << l << ", ";
            }
            std::ostringstream oss;
            oss << "In " << modelName << " network " << type << " layer '" << layer.first  << "' layout should be one of the following  "
                << ossLayouts.str() << " - but " << desc.getLayout() << " given";

            throw std::logic_error(oss.str());
        }
        ++i;
    }
}

template<class InputsDataMap, class OutputsDataMap>
void  ModelBase::checkInputsOutputs(const std::string& modelName, const std::vector<ModelBase::IOPattern>& modelIOPattern,
    const InputsDataMap& inputsInfo, const OutputsDataMap& outputsInfo) {
    //--------------------------- Check inputs blobs ------------------------------------------------------
    slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
    check(modelName, "input", modelIOPattern[0], inputsInfo);

    // --------------------------- Check output blobs -----------------------------------------------------
    slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
    check(modelName, "output", modelIOPattern[1], outputsInfo);
}
