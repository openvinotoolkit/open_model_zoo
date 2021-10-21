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

#include "models/jpeg_restoration_model.h"
#include "utils/ocv_common.hpp"
#include <utils/slog.hpp>
#include <inference_engine.hpp>
#include <iostream>

JPEGRestorationModel::JPEGRestorationModel(const std::string& modelFileName, const cv::Size& inputImgSize, bool _jpegCompression) :
    ImageModel(modelFileName, false) {
        netInputHeight = inputImgSize.height;
        netInputWidth = inputImgSize.width;
        jpegCompression = _jpegCompression;

}

void JPEGRestorationModel::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input blobs ------------------------------------------------------

    InferenceEngine::ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    if (inputShapes.size() != 1)
        throw std::runtime_error("The JPEG Restoration model wrapper supports topologies only with 1 input");
    inputsNames.push_back(inputShapes.begin()->first);
    InferenceEngine::SizeVector& inSizeVector = inputShapes.begin()->second;
    if (inSizeVector.size() != 4 || inSizeVector[0] != 1 || inSizeVector[1] != 3)
        throw std::runtime_error("3-channel 4-dimensional model's input is expected");

    // --------------------------- Prepare output blobs -----------------------------------------------------
    const InferenceEngine::OutputsDataMap& outputInfo = cnnNetwork.getOutputsInfo();
    if (outputInfo.size() != 1)
        throw std::runtime_error("The JPEG Restoration model wrapper supports topologies only with 1 output");

    outputsNames.push_back(outputInfo.begin()->first);
    InferenceEngine::Data& data = *outputInfo.begin()->second;
    data.setPrecision(InferenceEngine::Precision::FP32);
    const InferenceEngine::SizeVector& outSizeVector = data.getTensorDesc().getDims();
    if (outSizeVector.size() != 4 || outSizeVector[0] != 1 || outSizeVector[1] != 3)
        throw std::runtime_error("3-channel 4-dimensional model's output is expected");

    changeInputSize(cnnNetwork);
}

void JPEGRestorationModel::changeInputSize(InferenceEngine::CNNNetwork& cnnNetwork) {
    InferenceEngine::ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    InferenceEngine::SizeVector& inputDims = inputShapes.begin()->second;

    if (inputDims[2] % stride || inputDims[3] % stride)
        throw std::runtime_error("The shape of the model input must be divisible by stride");

    netInputHeight = static_cast<int>((netInputHeight + stride - 1) / stride) * stride;
    netInputWidth = static_cast<int>((netInputWidth + stride - 1) / stride) * stride;

    inputDims[0] = 1;
    inputDims[2] = netInputHeight;
    inputDims[3] = netInputWidth;

    cnnNetwork.reshape(inputShapes);
}

std::shared_ptr<InternalModelData> JPEGRestorationModel::preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) {
    cv::Mat image = inputData.asRef<ImageInputData>().inputImage;
    size_t h = image.rows;
    size_t w = image.cols;
    cv::Mat resizedImage;
    if (jpegCompression) {
        std::vector<uchar> encimg;
        std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, 40};
        cv::imencode(".jpg", image, encimg, params);
        image = cv::imdecode(cv::Mat(encimg), 3);
    }

    if (netInputHeight - stride < h && h <= netInputHeight
        && netInputWidth - stride < w && w <= netInputWidth) {
        int bottom = netInputHeight - h;
        int right = netInputWidth - w;
        cv::copyMakeBorder(image, resizedImage, 0, bottom, 0, right,
                           cv::BORDER_CONSTANT, 0);
    } else {
        slog::warn << "\tChosen model aspect ratio doesn't match image aspect ratio" << slog::endl;
        cv::resize(image, resizedImage, cv::Size(netInputWidth, netInputHeight));
    }
    InferenceEngine::Blob::Ptr frameBlob = request->GetBlob(inputsNames[0]);
    matToBlob(resizedImage, frameBlob);

    return std::make_shared<InternalImageModelData>(image.cols, image.rows);
}

std::unique_ptr<ResultBase> JPEGRestorationModel::postprocess(InferenceResult& infResult) {
    ImageResult* result = new ImageResult;
    *static_cast<ResultBase*>(result) = static_cast<ResultBase&>(infResult);

    const auto& inputImgSize = infResult.internalModelData->asRef<InternalImageModelData>();

    InferenceEngine::LockedMemory<const void> outMapped = infResult.getFirstOutputBlob()->rmap();
    const auto outputData = outMapped.as<float*>();

    std::vector<cv::Mat> imgPlanes;
    const InferenceEngine::SizeVector& outSizeVector = infResult.getFirstOutputBlob()->getTensorDesc().getDims();
    size_t outHeight = (int)(outSizeVector[2]);
    size_t outWidth = (int)(outSizeVector[3]);
    size_t numOfPixels = outWidth * outHeight;
    imgPlanes = std::vector<cv::Mat>{
          cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[0])),
          cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[numOfPixels])),
          cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[numOfPixels * 2]))};
    cv::Mat resultImg;
    cv::merge(imgPlanes, resultImg);

    if (netInputHeight - stride < static_cast<size_t>(inputImgSize.inputImgHeight) && static_cast<size_t>(inputImgSize.inputImgHeight) <= netInputHeight
        && netInputWidth - stride < static_cast<size_t>(inputImgSize.inputImgWidth) && static_cast<size_t>(inputImgSize.inputImgWidth) <= netInputWidth) {
        result->resultImage = resultImg(cv::Rect(0, 0, inputImgSize.inputImgWidth, inputImgSize.inputImgHeight));
    } else {
        cv::resize(resultImg, result->resultImage, cv::Size(inputImgSize.inputImgWidth, inputImgSize.inputImgHeight));
    }

    result->resultImage.convertTo(result->resultImage, CV_8UC3, 255);

    return std::unique_ptr<ResultBase>(result);
}
