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

#include "models/super_resolution_model.h"
#include "utils/ocv_common.hpp"
#include <utils/slog.hpp>

using namespace InferenceEngine;

SuperResolutionModel::SuperResolutionModel(const std::string& modelFileName, const cv::Size& inputImgSize) :
    ImageModel(modelFileName, false) {
        netInputHeight = inputImgSize.height;
        netInputWidth = inputImgSize.width;
}

void SuperResolutionModel::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output ---------------------------------------------
    // --------------------------- Prepare input blobs --------------------------------------------------

    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    if (inputShapes.size() != 1 && inputShapes.size() != 2)
        throw std::runtime_error("The demo supports topologies with 1 or 2 inputs only");
    std::string lrInputBlobName = inputShapes.begin()->first;
    inputsNames.push_back(lrInputBlobName);
    SizeVector& lrShape = inputShapes[lrInputBlobName];
    if (lrShape.size() != 4)
        throw std::runtime_error("Number of dimensions for an input must be 4");
    if (lrShape[1] != 1 && lrShape[1] != 3)
        throw std::runtime_error("Input layer is expected to have 1 or 3 channels");

    // A model like single-image-super-resolution-???? may take bicubic interpolation of the input image as the
    // second input
    std::string bicInputBlobName;
    if (inputShapes.size() == 2) {
        bicInputBlobName = (++inputShapes.begin())->first;
        inputsNames.push_back(bicInputBlobName);
        SizeVector& bicShape = inputShapes[bicInputBlobName];
        if (bicShape.size() != 4) {
            throw std::runtime_error("Number of dimensions for both inputs must be 4");
        }
        if (lrShape[2] >= bicShape[2] && lrShape[3] >= bicShape[3]) {
            inputsNames[0].swap(inputsNames[1]);
        } else if (!(lrShape[2] <= bicShape[2] && lrShape[3] <= bicShape[3])) {
            throw std::runtime_error("Each spatial dimension of one input must surpass or be equal to a spatial"
                "dimension of another input");
        }
    }

    InputInfo& inputInfo = *cnnNetwork.getInputsInfo().begin()->second;
    inputInfo.setPrecision(Precision::FP32);
    // --------------------------- Prepare output blobs -----------------------------------------------------
    const OutputsDataMap& outputInfo = cnnNetwork.getOutputsInfo();
    if (outputInfo.size() != 1)
        throw std::runtime_error("Demo supports topologies only with 1 output");

    outputsNames.push_back(outputInfo.begin()->first);
    Data& data = *outputInfo.begin()->second;
    data.setPrecision(Precision::FP32);
    changeInputSize(cnnNetwork, data.getDims()[2] / inputShapes[inputsNames[0]][2]);
}

void SuperResolutionModel::changeInputSize(CNNNetwork& cnnNetwork, int coeff) {
    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    SizeVector& lrShape = inputShapes[inputsNames[0]];

    lrShape[0] = 1;
    lrShape[2] = netInputHeight;
    lrShape[3] = netInputWidth;

    if (inputShapes.size() == 2) {
        SizeVector& bicShape = inputShapes[inputsNames[1]];
        bicShape[0] = 1;
        bicShape[2] = coeff * netInputHeight;
        bicShape[3] = coeff * netInputWidth;
    }
    cnnNetwork.reshape(inputShapes);
}

std::shared_ptr<InternalModelData> SuperResolutionModel::preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) {
    auto imgData = inputData.asRef<ImageInputData>();
    auto& img = imgData.inputImage;

    /* Resize and copy data from the image to the input blob */
    Blob::Ptr lrInputBlob = request->GetBlob(inputsNames[0]);
    if (img.channels() != (int)lrInputBlob->getTensorDesc().getDims()[1])
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    if (static_cast<size_t>(img.cols) != netInputWidth || static_cast<size_t>(img.rows) != netInputHeight)
        slog::warn << "Chosen model aspect ratio doesn't match image aspect ratio\n";
    matU8ToBlob<float_t>(img, lrInputBlob);

    if (inputsNames.size() == 2) {
        Blob::Ptr bicInputBlob = request->GetBlob(inputsNames[1]);

        int w = bicInputBlob->getTensorDesc().getDims()[3];
        int h = bicInputBlob->getTensorDesc().getDims()[2];
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(w, h), 0, 0, cv::INTER_CUBIC);
        matU8ToBlob<float_t>(resized, bicInputBlob);
    }

    return std::make_shared<InternalImageModelData>(img.cols, img.rows);
}

std::unique_ptr<ResultBase> SuperResolutionModel::postprocess(InferenceResult& infResult) {
    ImageResult* result = new ImageResult;
    *static_cast<ResultBase*>(result) = static_cast<ResultBase&>(infResult);


    LockedMemory<const void> outMapped = infResult.getFirstOutputBlob()->rmap();
    const auto outputData = outMapped.as<float*>();

    std::vector<cv::Mat> imgPlanes;
    const SizeVector& outSizeVector = infResult.getFirstOutputBlob()->getTensorDesc().getDims();
    size_t outChannels = (int)(outSizeVector[1]);
    size_t outHeight = (int)(outSizeVector[2]);
    size_t outWidth = (int)(outSizeVector[3]);
    size_t numOfPixels = outWidth * outHeight;
    if (outChannels == 3) {
        imgPlanes = std::vector<cv::Mat>{
              cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[0])),
              cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[numOfPixels])),
              cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[numOfPixels * 2]))};
    } else {
        imgPlanes = std::vector<cv::Mat>{cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[0]))};
        // Post-processing for text-image-super-resolution models
        cv::threshold(imgPlanes[0], imgPlanes[0], 0.5f, 1.0f, cv::THRESH_BINARY);
    }
    for (auto & img : imgPlanes)
        img.convertTo(img, CV_8UC1, 255);

    cv::Mat resultImg;
    cv::merge(imgPlanes, resultImg);
    result->resultImage = resultImg;

    return std::unique_ptr<ResultBase>(result);
}
