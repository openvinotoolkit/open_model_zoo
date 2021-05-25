/*
// Copyright (C) 2018-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
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
#include <utils/slog.hpp>

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
    const std::vector<std::string>& getInputsNames() const { return inputsNames; }

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

        BlobPattern(std::string&& type, std::map<size_t, std::map<std::string, InferenceEngine::TensorDesc>>&& patterns)
            : type(type), patterns(patterns) {}
    };

    using IOPattern = std::pair<std::string, std::vector<ModelBase::BlobPattern>>;
protected:
    virtual IOPattern getIOPattern() = 0;
    void prepareIECore(InferenceEngine::Core& core);

    template<class InputsDataMap, class OutputsDataMap>
    void findIONames(const IOPattern& modelBlobPattern, const InputsDataMap& inputsInfo, const OutputsDataMap& outputInfo);

    template<class DataMap>
    void findAndCheckBlobNames(const BlobPattern& pattern, std::vector<std::string>& names, const DataMap& info, const std::string& modelName);

    void prepareBlobs(const IOPattern& modelBlobPattern, const InferenceEngine::InputsDataMap& inputInfo, const InferenceEngine::OutputsDataMap& outputInfo);

    template<class InputsDataMap, class OutputsDataMap>
    void checkInputsOutputs(const IOPattern& modelBlobPattern,
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


template<class DataMap>
void ModelBase::findAndCheckBlobNames(const BlobPattern& pattern, std::vector<std::string>& names, const DataMap& info, const std::string& modelName) {
    //--------------------------- Find IO names --------------------------------
    //--------------------------- Check number of I/O---------------------------
    auto blobsNum = info.size();
    if (pattern.patterns.find(blobsNum) == pattern.patterns.end()) {
        std::ostringstream ossNums;
        for (const auto& n : pattern.patterns) {
            ossNums << n.first << ", ";
        }

        throw std::logic_error("Number of " + pattern.type + "s in " + modelName + " network should be one of the following : " + ossNums.str() +
            " - but " + std::to_string(blobsNum) + " given");
    }

    size_t blobsFound = 0;
    auto& blobsPatterns = pattern.patterns.find(blobsNum)->second;
    for (const auto& blob : info) {
        const auto& b = blobsPatterns.find(blob.first);
        if (b != blobsPatterns.end()) {
            names.push_back(blob.first);
            ++blobsFound;
        }
    }

    if (names.empty() && blobsNum == 1) {
        const auto& b = blobsPatterns.find("common");
        if (b != blobsPatterns.end()) {
            auto blobName = info.begin()->first;
            names.push_back(blobName);
            ++blobsFound;
        }
    }

    if (blobsFound == 0) {
        throw std::logic_error(modelName + " network has wrong " + pattern.type + " blobs' names");
    }
}

template<class InputsDataMap, class OutputsDataMap>
void ModelBase::findIONames(const IOPattern& modelBlobPattern, const InputsDataMap& inputInfo, const OutputsDataMap& outputInfo) {
    //--------------------------- Find input names ---------------------------
    findAndCheckBlobNames(modelBlobPattern.second[0], inputsNames, inputInfo, modelBlobPattern.first);

    //--------------------------- Find outputs names ---------------------------
    findAndCheckBlobNames(modelBlobPattern.second[1], outputsNames, outputInfo, modelBlobPattern.first);
}

template<class DataMap>
void check(const std::string& modelName, const ModelBase::BlobPattern& pattern,
    const DataMap& info, std::vector<std::string>& names) {
    // --------------------------- Check BLOB ---------------------------
    auto blobsNum = info.size();
    const auto& blobsPatterns = pattern.patterns.find(blobsNum)->second;
    for (auto& blobName : names) {
        auto blobPatternIt = blobsPatterns.find(blobName);
        if (blobPatternIt == blobsPatterns.end()) {
            blobPatternIt = blobsPatterns.find("common");
        }
        const auto& blobPatternDesc = blobPatternIt->second;
        const auto& blobPatternDims = blobPatternDesc.getDims();

        const auto& blobToCheckDesc = info.find(blobName)->second->getTensorDesc();
        const auto& blobToCheckDims = blobToCheckDesc.getDims();


        // --------------------------- Check number of dimensions ---------------------------
        if (blobToCheckDims.size() != blobPatternDims.size()) {
            throw std::logic_error("Unsupported " + std::to_string(blobToCheckDims.size()) + "D " + pattern.type + " blob '" + blobName + "'. " +
                " In " + modelName + " network number of dimensions for '" + blobName + "' blob should be : " + std::to_string(blobPatternDims.size()));
        }

        // --------------------------- Check dimensions -------------------------------------
        for (size_t i = 0; i < blobToCheckDims.size(); ++i) {
            if (blobPatternDims[i] != blobToCheckDims[i] && blobPatternDims[i] != 0) {
                std::ostringstream ossDims1, ossDims2;
                for (size_t i = 0; i < blobToCheckDims.size(); ++i) {
                    ossDims1 << blobToCheckDims[i] << ",";
                    if (blobPatternDims[i] == 0) {
                        ossDims2 << "X,";
                    }
                    else {
                        ossDims2 << blobPatternDims[i] << ",";
                    }
                }
                throw std::logic_error("Unsupported " + pattern.type + " blob '" + blobName + "' with dimension [" + ossDims1.str() + "]. " +
                    " In " + modelName + " network demensions for '" + blobName + "' blob should be : [" + ossDims2.str() + "]");
            }
        }

        // --------------------------- Check precision-------------------------------------
        if (blobPatternDesc.getPrecision() != blobToCheckDesc.getPrecision()) {
            throw std::logic_error("In " + modelName + " network " +
                pattern.type + " blob '" + blobName + "' precision should be: " + blobPatternDesc.getPrecision().name() +
                " - but " + blobToCheckDesc.getPrecision().name() + " given");
        }

        // Keep proper input names order
        if (blobToCheckDims.size() == 4 && blobToCheckDesc.getPrecision() == InferenceEngine::Precision::U8 &&
            names.size() > 1 && names[0] != blobName) {
            std::swap(names[0], names[1]);
        }

        // --------------------------- Check layout----------------------------------------
        if (blobPatternDesc.getLayout() != blobToCheckDesc.getLayout()) {
            std::ostringstream oss;
            oss << "In " << modelName << " network " << pattern.type << " blob '" << blobName << "' layout should be "
                << blobPatternDesc.getLayout() << " - but " << blobToCheckDesc.getLayout() << " given";

            throw std::logic_error(oss.str());
        }
    }
}


template<class InputsDataMap, class OutputsDataMap>
void ModelBase::checkInputsOutputs(const IOPattern& modelBlobPattern,
    const InputsDataMap& inputsInfo, const OutputsDataMap& outputsInfo) {

    //--------------------------- Check input blobs ------------------------------------------------------
    slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
    check(modelBlobPattern.first, modelBlobPattern.second[0], inputsInfo, inputsNames);

    // --------------------------- Check output blobs -----------------------------------------------------
    slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
    check(modelBlobPattern.first, modelBlobPattern.second[1], outputsInfo, outputsNames);
}
