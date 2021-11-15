/*
// Copyright (C) 2021 Intel Corporation
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

#include "models/landmarks_model.h"
#include <utils/common.hpp>

LandmarksModel::LandmarksModel(const std::string& modelFileName, bool useAutoResize, std::string postprocessType) :
    ImageModel(modelFileName, useAutoResize) {
    postprocessType = postprocessType;
}


void LandmarksModel::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input blobs ------------------------------------------------------
    InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error("Landmarks network should have only one input");
    }
    inputsNames.push_back(inputInfo.begin()->first);
    auto layerData = inputInfo.begin()->second;
    auto layerDims = layerData->getTensorDesc().getDims();
    const InferenceEngine::TensorDesc& inputDesc = layerData->getTensorDesc();
    netInputHeight = getTensorHeight(inputDesc);
    netInputWidth = getTensorWidth(inputDesc);

    if (layerDims.size() == 4) {
        layerData->setLayout(InferenceEngine::Layout::NCHW);
        layerData->setPrecision(InferenceEngine::Precision::U8);
    }
    else if (layerDims.size() == 2) {
        layerData->setLayout(InferenceEngine::Layout::NC);
        layerData->setPrecision(InferenceEngine::Precision::FP32);
    }
    else {
        throw std::runtime_error("Unknown type of input layer layout. Expected either 4 or 2 dimensional inputs");
    }
    // --------------------------- Prepare output blobs -----------------------------------------------------
    InferenceEngine::OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
    outputsNames.push_back(outputInfo.begin()->first);

    if (outputInfo.size() != 1) {
        throw std::logic_error("Landmarks network should have only one output");
    }
    InferenceEngine::Data& output = *outputInfo.begin()->second;
    output.setPrecision(InferenceEngine::Precision::FP32);
    const InferenceEngine::SizeVector& outSizeVector = output.getTensorDesc().getDims();
    if (outSizeVector.size() != 2 && outSizeVector.size() != 4) {
        throw std::logic_error("Landmarks Estimation network output layer should have 2 or 4 dimensions");
    }
    
}


std::shared_ptr<InternalModelData> LandmarksModel::preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) {
    
        const auto& origImg = inputData.asRef<ImageInputData>().inputImage;
        const auto& img = inputTransform(origImg);

        cv::Mat resizedImage;
        auto scaledSize = cv::Size(static_cast<int>(netInputWidth), static_cast<int>(netInputHeight));
        cv::resize(img, resizedImage, scaledSize, 0, 0, cv::INTER_CUBIC); 

        auto inputBlob = request->GetBlob(inputsNames[0]);
        matToBlob(resizedImage, inputBlob);

        frameHeight = img.cols;
        frameWidth = img.rows;
        return std::make_shared<InternalImageModelData>(img.cols, img.rows);

}

std::unique_ptr<ResultBase> LandmarksModel::postprocess(InferenceResult& infResult) {
    InferenceEngine::LockedMemory<const void> outputMapped = infResult.getFirstOutputBlob()->rmap();
    InferenceEngine::MemoryBlob::Ptr  output = infResult.getFirstOutputBlob();
    auto numberOfCoordinates = output->getTensorDesc().getDims()[1];
    auto normed_coordinates =output->rmap().as<float*>();

    LandmarksResult* result = new LandmarksResult(infResult.frameId, infResult.metaData);
    auto retVal = std::unique_ptr<ResultBase>(result);
    for (auto i = 0; i < numberOfCoordinates/2; ++i) {
        int normed_x = static_cast<int>(normed_coordinates[2 * i] * frameHeight);
        int normed_y = static_cast<int>(normed_coordinates[2*i + 1] * frameWidth);

        result->coordinates.push_back(cv::Point2f(normed_x, normed_y));
    }
    return retVal;
}

std::unique_ptr<ResultBase> LandmarksModel::simplePostprocess(InferenceResult& infResult) {
    InferenceEngine::LockedMemory<const void> outputMapped = infResult.getFirstOutputBlob()->rmap();
    InferenceEngine::MemoryBlob::Ptr  output = infResult.getFirstOutputBlob();
    auto numberOfCoordinates = output->getTensorDesc().getDims()[1];
    auto normed_coordinates = output->rmap().as<float*>();
    //auto normed_coordinates = outputMapped.as<float*>();

    LandmarksResult* result = new LandmarksResult(infResult.frameId, infResult.metaData);
    auto retVal = std::unique_ptr<ResultBase>(result);
    for (auto i = 0; i < numberOfCoordinates / 2; ++i) {
        int normed_x = static_cast<int>(normed_coordinates[2 * i] * frameHeight);
        int normed_y = static_cast<int>(normed_coordinates[2 * i + 1] * frameWidth);

        result->coordinates.push_back(cv::Point2f(normed_x, normed_y));
    }
    return retVal;
}

/*std::unique_ptr<ResultBase> LandmarksModel::heatmapPostprocess(InferenceResult& infResult) {
    InferenceEngine::MemoryBlob::Ptr  outputMapped = infResult.getFirstOutputBlob();

    auto N = outputMapped->getTensorDesc().getDims()[0];
    auto K = outputMapped->getTensorDesc().getDims()[1];
    auto H = outputMapped->getTensorDesc().getDims()[2];
    auto W = outputMapped->getTensorDesc().getDims()[3];

}*/
