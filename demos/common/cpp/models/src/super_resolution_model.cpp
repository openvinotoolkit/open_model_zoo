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

#include "models/super_resolution_model.h"
#include "utils/ocv_common.hpp"

using namespace InferenceEngine;

SuperResolutionModel::SuperResolutionModel(const std::string& modelFileName) :
    ModelBase(modelFileName) {
}

void SuperResolutionModel::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output ---------------------------------------------
    // --------------------------- Prepare input blobs --------------------------------------------------
    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    if (inputShapes.size() != 1 && inputShapes.size() != 2)
        throw std::logic_error("The demo supports topologies with 1 or 2 inputs only");
    std::string firstInputBlobName = inputShapes.begin()->first;
    inputsNames.push_back(firstInputBlobName);
    SizeVector& firstInputSizeVector = inputShapes[firstInputBlobName];
    if (firstInputSizeVector.size() != 4)
        throw std::logic_error("Number of dimensions for an input must be 4");
    
    // A model like single-image-super-resolution-???? may take bicubic interpolation of the input image as the
    // second input
    std::string secondInputBlobName;
    if (inputShapes.size() == 2) {
        secondInputBlobName = (++inputShapes.begin())->first;
        inputsNames.push_back(secondInputBlobName);
        SizeVector& secondInputSizeVector = inputShapes[secondInputBlobName];
        if (secondInputSizeVector.size() != 4) {
            throw std::logic_error("Number of dimensions for both inputs must be 4");
        }
        if (firstInputSizeVector[2] >= secondInputSizeVector[2] && firstInputSizeVector[3] >= secondInputSizeVector[3]) {
            firstInputBlobName.swap(secondInputBlobName);
            firstInputSizeVector.swap(secondInputSizeVector);
        } else if (!(firstInputSizeVector[2] <= secondInputSizeVector[2] && firstInputSizeVector[3] <= secondInputSizeVector[3])) {
            throw std::logic_error("Each spatial dimension of one input must surpass or be equal to a spatial"
                "dimension of another input");
        }
    }

    InputInfo& inputInfo = *cnnNetwork.getInputsInfo().begin()->second;
    inputInfo.setPrecision(Precision::FP32);
    // --------------------------- Prepare output blobs -----------------------------------------------------
    const OutputsDataMap& outputInfo = cnnNetwork.getOutputsInfo();
    if (outputInfo.size() != 1) throw std::runtime_error("Demo supports topologies only with 1 output");

    outputsNames.push_back(outputInfo.begin()->first);
    Data& data = *outputInfo.begin()->second;
    data.setPrecision(Precision::FP32);
    const SizeVector& outSizeVector = data.getTensorDesc().getDims();
    outChannels = (int)(outSizeVector[1]);
    outHeight = (int)(outSizeVector[2]);
    outWidth = (int)(outSizeVector[3]);
}

std::shared_ptr<InternalModelData> SuperResolutionModel::preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) {
    auto imgData = inputData.asRef<ImageInputData>();
    auto& img = imgData.inputImage;

    /* Resize and copy data from the image to the input blob */
    Blob::Ptr lrInputBlob = request->GetBlob(inputsNames[0]);
    matU8ToBlob<float_t>(img, lrInputBlob);

    if (inputsNames.size() == 2) {
        Blob::Ptr bicInputBlob = request->GetBlob(inputsNames[1]);

        int w = bicInputBlob->getTensorDesc().getDims()[3];
        int h = bicInputBlob->getTensorDesc().getDims()[2];
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(w, h), 0, 0, cv::INTER_CUBIC);
        matU8ToBlob<float_t>(resized, bicInputBlob);
    }

    return std::shared_ptr<InternalModelData>(new InternalImageModelData(img.cols, img.rows));
}

std::unique_ptr<ResultBase> SuperResolutionModel::postprocess(InferenceResult& infResult) {
    ImageProcessingResult* result = new ImageProcessingResult;
    *static_cast<ResultBase*>(result) = static_cast<ResultBase&>(infResult);

    LockedMemory<const void> outMapped = infResult.getFirstOutputBlob()->rmap();
    const auto outputData = outMapped.as<float*>();

    std::vector<cv::Mat> imgPlanes;
    size_t numOfPixels = outWidth * outHeight;
    if (outChannels == 3) {
        imgPlanes = std::vector<cv::Mat>{
              cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[0])),
              cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[numOfPixels])),
              cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[numOfPixels * 2]))};
    } else {
        imgPlanes = std::vector<cv::Mat>{cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[numOfPixels * outChannels]))};
        // Post-processing for text-image-super-resolution models
        cv::threshold(imgPlanes[0], imgPlanes[0], 0.5f, 1.0f, cv::THRESH_BINARY);
    }
    for (auto & img : imgPlanes)
        img.convertTo(img, CV_8UC1, 255);

    cv::merge(imgPlanes, result->resultImage);
    return std::unique_ptr<ResultBase>(result);
}
