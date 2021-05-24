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

#include "models/detection_model_ssd.h"
#include <utils/slog.hpp>
#include <utils/common.hpp>
#include <ngraph/ngraph.hpp>

ModelSSD::ModelSSD(const std::string& modelFileName,
    float confidenceThreshold, bool useAutoResize,
    const std::vector<std::string>& labels) :
    DetectionModel(modelFileName, confidenceThreshold, useAutoResize, labels) {
}

ModelBase::IOPattern ModelSSD::getIOPattern() {
    ModelBase::BlobPattern inputPattern(
        "input",
        // Possible models' inputs
        // Describe number of inputs, precision, dimensions and layout.
        // If it doesn't matter what dimension's value is set 0.
        {
            { 1, {  { "common", { InferenceEngine::Precision::U8, {1, 3, 0, 0}, useAutoResize ? InferenceEngine::Layout::NHWC : InferenceEngine::Layout::NCHW} } } } ,

            { 2, {  { "image_tensor", {  InferenceEngine::Precision::U8, {1, 3, 0, 0}, InferenceEngine::Layout::NCHW } },
                    { "image_info", {  InferenceEngine::Precision::FP32, {1, 3}, InferenceEngine::Layout::NC } } } },

        }
    );

    ModelBase::BlobPattern outputPattern(
        "output",
        // Possible models' outputs
        // Describe number of inputs, precision, dimensions and layout.
        // If it doesn't matter what dimension's value is - set 0.
        {
            { 1, {  { "common", { InferenceEngine::Precision::FP32, {1, 1, 0, 7}, InferenceEngine::Layout::NCHW} } } },

            { 4, {  { "bboxes", { InferenceEngine::Precision::FP32, {1, 0, 4}, InferenceEngine::Layout::CHW } },
                    { "labels", { InferenceEngine::Precision::FP32, {1, 200}, InferenceEngine::Layout::NC } },
                    { "scores", { InferenceEngine::Precision::FP32, {1, 200}, InferenceEngine::Layout::NC } } } },

            { 8, {  { "boxes", { InferenceEngine::Precision::FP32, {0, 5}, InferenceEngine::Layout::NC } },
                    { "labels", { InferenceEngine::Precision::FP32, {100}, InferenceEngine::Layout::C } } } },

            { 9, {  { "boxes", { InferenceEngine::Precision::FP32, {0, 5}, InferenceEngine::Layout::NC } },
                    { "labels", { InferenceEngine::Precision::FP32, {100}, InferenceEngine::Layout::C } } } },
        }
    );

    return { "SSD", {inputPattern, outputPattern} };
}

template<class OutputsDataMap>
void ModelSSD::getBlobDims(const OutputsDataMap& outputInfo) {
    const InferenceEngine::SizeVector outputDims = outputInfo.find(outputsNames[0])->second->getTensorDesc().getDims();
    auto nDims = outputDims.size();
    maxProposalCount = outputDims[nDims - 2];
    objectSize = outputDims[nDims - 1];
}

void ModelSSD::checkCompiledNetworkInputsOutputs() {
    ImageModel::checkCompiledNetworkInputsOutputs();
    getBlobDims(execNetwork.GetOutputsInfo());
}

void ModelSSD::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output -------------------------------------------------
    ImageModel::prepareInputsOutputs(cnnNetwork);
    getBlobDims(cnnNetwork.getOutputsInfo());

}

std::shared_ptr<InternalModelData> ModelSSD::preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) {
    if (inputsNames.size() > 1) {
        auto blob = request->GetBlob(inputsNames[1]);
        InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->wmap();
        auto data = blobMapped.as<float*>();
        data[0] = static_cast<float>(netInputHeight);
        data[1] = static_cast<float>(netInputWidth);
        data[2] = 1;
    }

    return DetectionModel::preprocess(inputData, request);
}

std::unique_ptr<ResultBase> ModelSSD::postprocess(InferenceResult& infResult) {
    return outputsNames.size() > 1 ?
        postprocessMultipleOutputs(infResult) :
        postprocessSingleOutput(infResult);
}

std::unique_ptr<ResultBase> ModelSSD::postprocessSingleOutput(InferenceResult& infResult) {
    InferenceEngine::LockedMemory<const void> outputMapped = infResult.getFirstOutputBlob()->rmap();
    const float *detections = outputMapped.as<float*>();

    DetectionResult* result = new DetectionResult(infResult.frameId, infResult.metaData);
    auto retVal = std::unique_ptr<ResultBase>(result);

    const auto& internalData = infResult.internalModelData->asRef<InternalImageModelData>();

    for (size_t i = 0; i < maxProposalCount; i++) {
        float image_id = detections[i * objectSize + 0];
        if (image_id < 0) {
            break;
        }

        float confidence = detections[i * objectSize + 2];

        /** Filtering out objects with confidence < confidence_threshold probability **/
        if (confidence > confidenceThreshold) {
            DetectedObject desc;

            desc.confidence = confidence;
            desc.labelID = static_cast<int>(detections[i * objectSize + 1]);
            desc.label = getLabelName(desc.labelID);
            desc.x = detections[i * objectSize + 3] * internalData.inputImgWidth;
            desc.y = detections[i * objectSize + 4] * internalData.inputImgHeight;
            desc.width = detections[i * objectSize + 5] * internalData.inputImgWidth - desc.x;
            desc.height = detections[i * objectSize + 6] * internalData.inputImgHeight - desc.y;

            result->objects.push_back(desc);
        }
    }

    return retVal;
}

std::unique_ptr<ResultBase> ModelSSD::postprocessMultipleOutputs(InferenceResult& infResult) {
    std::vector<InferenceEngine::LockedMemory<const void>> mappedMemoryAreas;
    for (const auto& name : outputsNames) {
        mappedMemoryAreas.push_back(infResult.outputsData[name]->rmap());
    }

    const float *boxes = mappedMemoryAreas[0].as<float*>();
    const float *labels = mappedMemoryAreas[1].as<float*>();
    const float *scores = mappedMemoryAreas.size() > 2 ? mappedMemoryAreas[2].as<float*>() : nullptr;

    DetectionResult* result = new DetectionResult(infResult.frameId, infResult.metaData);
    auto retVal = std::unique_ptr<ResultBase>(result);

    const auto& internalData = infResult.internalModelData->asRef<InternalImageModelData>();

    // In models with scores are stored in separate output, coordinates are normalized to [0,1]
    // In other multiple-outputs models coordinates are normalized to [0,netInputWidth] and [0,netInputHeight]
    float widthScale = ((float)internalData.inputImgWidth) / (scores ? 1 : netInputWidth);
    float heightScale = ((float)internalData.inputImgHeight) / (scores ? 1 : netInputHeight);

    for (size_t i = 0; i < maxProposalCount; i++) {
        float confidence = scores ? scores[i] : boxes[i * objectSize + 4];

        /** Filtering out objects with confidence < confidence_threshold probability **/
        if (confidence > confidenceThreshold) {
            DetectedObject desc;

            desc.confidence = confidence;
            desc.labelID = static_cast<int>(labels[i]);
            desc.label = getLabelName(desc.labelID);
            desc.x = boxes[i * objectSize] * widthScale;
            desc.y = boxes[i * objectSize + 1] * heightScale;
            desc.width = boxes[i * objectSize + 2] * widthScale - desc.x;
            desc.height = boxes[i * objectSize + 3] * heightScale - desc.y;

            result->objects.push_back(desc);
        }
    }

    return retVal;
}
