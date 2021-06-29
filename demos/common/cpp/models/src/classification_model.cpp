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

#include "models/classification_model.h"
#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>

using namespace InferenceEngine;

ClassificationModel::ClassificationModel(const std::string& modelFileName, size_t nTop, bool useAutoResize, const std::vector<std::string>& labels) :
    ImageModel(modelFileName, useAutoResize),
    nTop(nTop),
    labels(labels) {
}

std::unique_ptr<ResultBase> ClassificationModel::postprocess(InferenceResult& infResult) {
    InferenceEngine::LockedMemory<const void> outputMapped = infResult.getFirstOutputBlob()->rmap();
    const float *classificationData = outputMapped.as<float*>();

    ClassificationResult* result = new ClassificationResult(infResult.frameId, infResult.metaData);
    auto retVal = std::unique_ptr<ResultBase>(result);

    std::vector<unsigned> indices(infResult.getFirstOutputBlob()->size());
    std::iota(std::begin(indices), std::end(indices), 0);
    std::partial_sort(std::begin(indices), std::begin(indices) + nTop, std::end(indices),
                    [&classificationData](unsigned l, unsigned r) {
                        return classificationData[l] > classificationData[r];
                    });
    result->topLabels.reserve(nTop);
    for (size_t i = 0; i < nTop; ++i) {
        result->topLabels.emplace_back(indices[i], labels[indices[i]]);
    }
    return retVal;
}

std::vector<std::string> ClassificationModel::loadLabels(const std::string& labelFilename) {
    std::vector<std::string> labels;

    /* Read labels */
    std::ifstream inputFile(labelFilename);
    if (!inputFile.is_open())
        throw std::runtime_error("Can't open the labels file: " + labelFilename);
    std::string labelsLine;
    while (std::getline(inputFile, labelsLine)) {
        size_t labelBeginIdx = labelsLine.find(' ');
        size_t labelEndIdx = labelsLine.find(',');  // can be npos when class has only one label
        if (labelBeginIdx == std::string::npos) {
            throw std::runtime_error("The labels file has incorrect format.");
        }
        labels.push_back(labelsLine.substr(labelBeginIdx + 1, labelEndIdx - (labelBeginIdx + 1)));
    }
    if (labels.empty())
        throw std::logic_error("File is empty: " + labelFilename);

    return labels;
}

void ClassificationModel::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input blobs ------------------------------------------------------
    InferenceEngine::ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    if (inputShapes.size() != 1)
        throw std::runtime_error("Demo supports topologies only with 1 input");
    inputsNames.push_back(inputShapes.begin()->first);
    SizeVector& inSizeVector = inputShapes.begin()->second;
    if (inSizeVector.size() != 4 || inSizeVector[1] != 3)
        throw std::runtime_error("3-channel 4-dimensional model's input is expected");
    if (inSizeVector[2] != inSizeVector[3])
        throw std::logic_error("Model input has incorrect image shape. Must be NxN square."
                                " Got " + std::to_string(inSizeVector[2]) +
                                "x" + std::to_string(inSizeVector[3]) + ".");

    InputInfo& inputInfo = *cnnNetwork.getInputsInfo().begin()->second;
    inputInfo.setPrecision(Precision::U8);
    if (useAutoResize) {
        inputInfo.getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
        inputInfo.getInputData()->setLayout(Layout::NHWC);
    }
    else
        inputInfo.getInputData()->setLayout(Layout::NCHW);

    // --------------------------- Prepare output blobs -----------------------------------------------------
    const OutputsDataMap& outputsDataMap = cnnNetwork.getOutputsInfo();
    if (outputsDataMap.size() != 1) throw std::runtime_error("Demo supports topologies only with 1 output");

    outputsNames.push_back(outputsDataMap.begin()->first);
    Data& data = *outputsDataMap.begin()->second;
    const SizeVector& outSizeVector = data.getTensorDesc().getDims();
    if (outSizeVector.size() != 2 && outSizeVector.size() != 4)
        throw std::runtime_error("Demo supports topologies only with 2-dimensional or 4-dimensional output");
    if (outSizeVector.size() == 4 && outSizeVector[2] != 1 && outSizeVector[3] != 1)
        throw std::runtime_error("Demo supports topologies only with 4-dimensional output which has last two dimensions of size 1");
    if (nTop > outSizeVector[1])
        throw std::runtime_error("The model provides " + std::to_string(outSizeVector[1]) + " classes, but " + std::to_string(nTop) + " labels are requested to be predicted");
    if (outSizeVector[1] == labels.size() + 1) {
        labels.insert(labels.begin(), "other");
        slog::warn << "Inserted 'other' label as first.\n";
    }
    else if (outSizeVector[1] != labels.size())
        throw std::logic_error("Model's number of classes and parsed labels must match (" + std::to_string(outSizeVector[1]) + " and " + std::to_string(labels.size()) + ')');

    data.setPrecision(Precision::FP32);
}
