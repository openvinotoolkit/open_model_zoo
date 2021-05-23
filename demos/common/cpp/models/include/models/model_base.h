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
#include <utils/config_factory.h>

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

    struct BlobPattern {
        const std::string type;
        const std::map<size_t, std::map<std::string, InferenceEngine::TensorDesc>> patterns;
        const InferenceEngine::ResizeAlgorithm resizeAlgo;

        BlobPattern(std::string&& type, std::map<size_t, std::map<std::string, InferenceEngine::TensorDesc>>&& patterns,
            InferenceEngine::ResizeAlgorithm&& algo = InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR)
            : type(std::move(type)), patterns(std::move(patterns)), resizeAlgo(algo) {}
    };

    using IOPattern = std::pair<std::string, std::vector<ModelBase::BlobPattern>>;
protected:
    IOPattern virtual getIOPattern() = 0;
    InferenceEngine::CNNNetwork prepareNetwork(InferenceEngine::Core& core);

    template<class InputsDataMap, class OutputsDataMap>
    void findIONames(const IOPattern& modelBlobPattern, const InputsDataMap& inputsInfo, const OutputsDataMap& outputInfo);

    template<class DataMap>
    void findAndCheckBlobNames(std::vector<std::string>& names, std::string modelName,
        const BlobPattern& pattern, const DataMap& info);

    void prepareBlobs(const IOPattern& modelBlobPattern, const InferenceEngine::InputsDataMap& inputInfo, const InferenceEngine::OutputsDataMap& outputInfo);

    template<class InputsDataMap, class OutputsDataMap>
    void checkInputsOutputs(const IOPattern& modelBlobPattern,
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


template<class DataMap>
void ModelBase::findAndCheckBlobNames(std::vector<std::string>& names, std::string modelName, const ModelBase::BlobPattern& pattern, const DataMap& info) {
    //--------------------------- Find IO names -------------------------------- 
    //--------------------------- Check number of I/O--------------------------- 
    auto blobsNum = info.size();
    if (pattern.patterns.find(blobsNum) == pattern.patterns.end()) {
        std::ostringstream ossNums;
        // Convert set to string
        for (const auto& n : pattern.patterns) {
            ossNums << n.first << ", ";
        }

        throw std::logic_error("Number of " + pattern.type + "s in " + modelName + " network should be one of the following : " + ossNums.str() +
            " - but " + std::to_string(blobsNum) + " given");
    }

    size_t blobsFound = 0;
    auto& blobsPatterns = pattern.patterns.find(blobsNum)->second;
    for (const auto& blob : info) {
        if (blobsNum > 1) {
            const auto& b = blobsPatterns.find(blob.first);
            if (b != blobsPatterns.end()) {
                names.push_back(blob.first);
                ++blobsFound;
            }
        } else {
            names.push_back(blob.first);
            ++blobsFound;
        }
    }

    if (blobsFound != blobsPatterns.size()) {
        throw std::logic_error(modelName + " network has wrong " + pattern.type + " layers' names");
    }
}

template<class InputsDataMap, class OutputsDataMap>
void ModelBase::findIONames(const IOPattern& modelBlobPattern, const InputsDataMap& inputInfo, const OutputsDataMap& outputInfo) {
    //--------------------------- Find input names --------------------------- 
    findAndCheckBlobNames(inputsNames, modelBlobPattern.first, modelBlobPattern.second[0], inputInfo);

    //--------------------------- Find outputs names --------------------------- 
    findAndCheckBlobNames(outputsNames, modelBlobPattern.first, modelBlobPattern.second[1], outputInfo);
}

template<class DataMap>
void check(const std::string& modelName, const ModelBase::BlobPattern& pattern,
    const DataMap& info, std::vector<std::string>& names) {
    // --------------------------- Check BLOB --------------------------- 
    auto& layersPatterns = pattern.patterns.find(info.size())->second;
    if (layersPatterns.begin()->first == "") {

    }
    for (auto& layerName : names) {
        const auto& layerPatternDesc = layersPatterns.begin()->first == "" ? layersPatterns.find(layerName)->second : layersPatterns.begin()->second;
        const auto& layerPatternDims = layerPatternDesc.getDims();

        const auto& layerToCheckDesc = info.find(layerName)->second->getTensorDesc();
        const auto& layerToCheckDims = layerToCheckDesc.getDims();


        // --------------------------- Check number of dimensions ---------------------------
        if (layerToCheckDims.size() != layerPatternDims.size()) {
            throw std::logic_error("Unsupported " + std::to_string(layerToCheckDims.size()) + "D " + pattern.type + " layer '" + layerName + "'. " +
                " In " + modelName + " network number of dimensions for '" + layerName + "' layer should be : " + std::to_string(layerPatternDims.size()));
        }

        // --------------------------- Check dimensions -------------------------------------
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
                throw std::logic_error("Unsupported " + pattern.type + " layer '" + layerName + "' with dimension [" + ossDims1.str() + "]. " +
                    " In " + modelName + " network demensions for '" + layerName + "' layer should be : [" + ossDims2.str() + "]");
            }
        }

        // --------------------------- Check precision-------------------------------------
        if (layerPatternDesc.getPrecision() != layerToCheckDesc.getPrecision()) {

            throw std::logic_error("In " + modelName + " network " +
                pattern.type + " layer '" + layerName + "' precision should be: " + layerPatternDesc.getPrecision().name() +
                " - but " + layerToCheckDesc.getPrecision().name() + " given");
        }

        // keep proper input order
        if (layerToCheckDims.size() == 4 && layerToCheckDesc.getPrecision() == InferenceEngine::Precision::U8 &&
            names.size() > 1 && names[0] != layerName) {
            std::swap(names[0], names[1]);
        }

        // --------------------------- Check layout----------------------------------------
        if (layerPatternDesc.getLayout() != layerToCheckDesc.getLayout()) {
            std::ostringstream oss;
            oss << "In " << modelName << " network " << pattern.type << " layer '" << layerName << "' layout should be "
                << layerPatternDesc.getLayout() << " - but " << layerToCheckDesc.getLayout() << " given";

            throw std::logic_error(oss.str());
        }
    }
}


template<class InputsDataMap, class OutputsDataMap>
void  ModelBase::checkInputsOutputs(const IOPattern& modelBlobPattern,
    const InputsDataMap& inputsInfo, const OutputsDataMap& outputsInfo) {


    ////--------------------------- Check inputs blobs ------------------------------------------------------
    //slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
    //fillCheckBlobNames(inputsNames, modelBlobPattern.first, modelBlobPattern.second[0], inputsInfo);

    //// --------------------------- Check output blobs -----------------------------------------------------
    //slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
    //fillCheckBlobNames(outputsNames, modelBlobPattern.first, modelBlobPattern.second[1], outputsInfo);

    //prepareInputsOutputs_(modelBlobPattern.second, inputsInfo,  outputsInfo);
    //--------------------------- Check inputs blobs ------------------------------------------------------
    slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
    check(modelBlobPattern.first, modelBlobPattern.second[0], inputsInfo, inputsNames);

    // --------------------------- Check output blobs -----------------------------------------------------
    slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
    check(modelBlobPattern.first, modelBlobPattern.second[1], outputsInfo, outputsNames);
}
