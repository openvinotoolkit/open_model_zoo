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

#include "models/style_transfer_model.h"

#include "utils/ocv_common.hpp"
#include <utils/slog.hpp>

#include <string>
#include <vector>
#include <memory>

using namespace InferenceEngine;

StyleTransferModel::StyleTransferModel(const std::string& modelFileName) :
    ImageModel(modelFileName, false) {
}

template<typename T>
void tell_me_type(T t, std::string name) {
    std::cout << name << " " << __PRETTY_FUNCTION__ << std::endl;
}

void StyleTransferModel::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output ---------------------------------------------
    // --------------------------- Prepare input blobs --------------------------------------------------

    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    if (inputShapes.size() != 1)
        throw std::runtime_error("Demo supports topologies only with 1 input");
    inputsNames.push_back(inputShapes.begin()->first);
    SizeVector& inSizeVector = inputShapes.begin()->second;
    if (inSizeVector.size() != 4 || inSizeVector[0] != 1 || inSizeVector[1] != 3)
        throw std::runtime_error("3-channel 4-dimensional model's input is expected");
    InputInfo& inputInfo = *cnnNetwork.getInputsInfo().begin()->second;
    inputInfo.setPrecision(Precision::FP32);

    // --------------------------- Prepare output blobs -----------------------------------------------------
    const OutputsDataMap& outputInfo = cnnNetwork.getOutputsInfo();
    if (outputInfo.size() != 1)
        throw std::runtime_error("Demo supports topologies only with 1 output");

    outputsNames.push_back(outputInfo.begin()->first);
    Data& data = *outputInfo.begin()->second;
    data.setPrecision(Precision::FP32);
    const SizeVector& outSizeVector = data.getTensorDesc().getDims();
    if (outSizeVector.size() != 4 || outSizeVector[0] != 1 || outSizeVector[1] != 3)
        throw std::runtime_error("3-channel 4-dimensional model's output is expected");

}

std::shared_ptr<InternalModelData> StyleTransferModel::preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) {
    auto imgData = inputData.asRef<ImageInputData>();
    auto& img = imgData.inputImage;

    Blob::Ptr minput = request->GetBlob(inputsNames[0]);
    matToBlob(img, minput);
    return std::make_shared<InternalImageModelData>(img.cols, img.rows);
}

std::unique_ptr<ResultBase> StyleTransferModel::postprocess(InferenceResult& infResult) {

    ImageResult* result = new ImageResult;
    *static_cast<ResultBase*>(result) = static_cast<ResultBase&>(infResult);

    const auto& inputImgSize = infResult.internalModelData->asRef<InternalImageModelData>();

    LockedMemory<const void> outMapped = infResult.getFirstOutputBlob()->rmap();
    const auto outputData = outMapped.as<float*>();

    const SizeVector& outSizeVector = infResult.getFirstOutputBlob()->getTensorDesc().getDims();
    size_t outHeight = (int)(outSizeVector[2]);
    size_t outWidth = (int)(outSizeVector[3]);
    size_t numOfPixels = outWidth * outHeight;

    std::vector<cv::Mat> imgPlanes;
    imgPlanes = std::vector<cv::Mat>{
              cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[numOfPixels * 2])),
              cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[numOfPixels])),
              cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[0]))};
    cv::Mat resultImg;
    cv::merge(imgPlanes, resultImg);
    cv::resize(resultImg, result->resultImage, cv::Size(inputImgSize.inputImgWidth, inputImgSize.inputImgHeight));

    result->resultImage.convertTo(result->resultImage, CV_8UC3);

    return std::unique_ptr<ResultBase>(result);
}
