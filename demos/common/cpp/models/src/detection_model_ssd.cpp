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

using namespace InferenceEngine;

ModelSSD::ModelSSD(const std::string& modelFileName,
    float confidenceThreshold, bool useAutoResize,
    const std::vector<std::string>& labels) :
    DetectionModel(modelFileName, confidenceThreshold, useAutoResize, labels) {
}

std::shared_ptr<InternalModelData> ModelSSD::preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) {
    if (inputsNames.size() > 1) {
        auto blob = request->GetBlob(inputsNames[1]);
        LockedMemory<void> blobMapped = as<MemoryBlob>(blob)->wmap();
        auto data = blobMapped.as<float*>();
        data[0] = static_cast<float>(netInputHeight);
        data[1] = static_cast<float>(netInputWidth);
        data[2] = 1;
    }

    return DetectionModel::preprocess(inputData, request);
}

std::unique_ptr<ResultBase> ModelSSD::postprocess(InferenceResult& infResult) {
    LockedMemory<const void> outputMapped = infResult.getFirstOutputBlob()->rmap();
    const float *detections = outputMapped.as<float*>();

    DetectionResult* result = new DetectionResult;
    auto retVal = std::unique_ptr<ResultBase>(result);

    *static_cast<ResultBase*>(result) = static_cast<ResultBase&>(infResult);

    const auto& internalData = infResult.internalModelData->asRef<InternalImageModelData>();

    for (size_t i = 0; i < maxProposalCount; i++) {
        float image_id = detections[i * objectSize + 0];
        if (image_id < 0) {
            break;
        }

        float confidence = detections[i * objectSize + 2];
        if (confidence > confidenceThreshold) {
            DetectedObject desc;

            desc.confidence = confidence;
            desc.labelID = static_cast<int>(detections[i * objectSize + 1]);
            desc.label = getLabelName(desc.labelID);
            desc.x = detections[i * objectSize + 3] * internalData.inputImgWidth;
            desc.y = detections[i * objectSize + 4] * internalData.inputImgHeight;
            desc.width = detections[i * objectSize + 5] * internalData.inputImgWidth - desc.x;
            desc.height = detections[i * objectSize + 6] * internalData.inputImgHeight - desc.y;

            /** Filtering out objects with confidence < confidence_threshold probability **/
            result->objects.push_back(desc);
        }
    }

    return retVal;
}

void ModelSSD::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input blobs ------------------------------------------------------
    slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());

    for (const auto& inputInfoItem : inputInfo) {
        if (inputInfoItem.second->getTensorDesc().getDims().size() == 4) {  // 1st input contains images
            if (inputsNames.empty()) {
                inputsNames.push_back(inputInfoItem.first);
            }
            else {
                inputsNames[0] = inputInfoItem.first;
            }

            inputInfoItem.second->setPrecision(Precision::U8);
            if (useAutoResize) {
                inputInfoItem.second->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
                inputInfoItem.second->getInputData()->setLayout(Layout::NHWC);
            }
            else {
                inputInfoItem.second->getInputData()->setLayout(Layout::NCHW);
            }
            const TensorDesc& inputDesc = inputInfoItem.second->getTensorDesc();
            netInputHeight = getTensorHeight(inputDesc);
            netInputWidth = getTensorWidth(inputDesc);
        }
        else if (inputInfoItem.second->getTensorDesc().getDims().size() == 2) {  // 2nd input contains image info
            inputsNames.resize(2);
            inputsNames[1] = inputInfoItem.first;
            inputInfoItem.second->setPrecision(Precision::FP32);
        }
        else {
            throw std::logic_error("Unsupported " +
                std::to_string(inputInfoItem.second->getTensorDesc().getDims().size()) + "D "
                "input layer '" + inputInfoItem.first + "'. "
                "Only 2D and 4D input layers are supported");
        }
    }

    // --------------------------- Prepare output blobs -----------------------------------------------------
    slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
    OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
    if (outputInfo.size() != 1) {
        throw std::logic_error("This demo accepts networks having only one output");
    }
    DataPtr& output = outputInfo.begin()->second;
    outputsNames.push_back(outputInfo.begin()->first);

    const SizeVector outputDims = output->getTensorDesc().getDims();

    if (outputDims.size() != 4) {
        throw std::logic_error("Incorrect output dimensions for SSD");
    }

    maxProposalCount = outputDims[2];
    objectSize = outputDims[3];
    if (objectSize != 7) {
        throw std::logic_error("Output should have 7 as a last dimension");
    }

    output->setPrecision(Precision::FP32);
    output->setLayout(Layout::NCHW);
}
