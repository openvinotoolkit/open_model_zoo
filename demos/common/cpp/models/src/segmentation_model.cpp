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

#include "models/segmentation_model.h"
#include <utils/slog.hpp>
#include <utils/ocv_common.hpp>

SegmentationModel::SegmentationModel(const std::string& modelFileName, bool useAutoResize) :
    ImageModel(modelFileName, useAutoResize) {
}

ModelBase::IOPattern SegmentationModel::getIOPattern() {
    ModelBase::BlobPattern inputPattern(
        "input",
        // Possible models' inputs
        // Describe number of inputs, precision, dimensions and layout.
        // If it doesn't matter what dimension's value is set 0.
        {
            { 1, {  { "common", { InferenceEngine::Precision::U8, {1, 3, 0, 0}, useAutoResize ? InferenceEngine::Layout::NHWC : InferenceEngine::Layout::NCHW} } } } ,
        }
    );

    ModelBase::BlobPattern outputPattern(
        "output",
        // Possible models' outputs
        // Describe number of outputs, precision, dimensions and layout.
        // If it doesn't matter what dimension's value is - set 0.
        {
            { 1, {  { "common", { InferenceEngine::Precision::FP32, {1, 0, 1024, 2048}, InferenceEngine::Layout::NCHW} },
                    { "ArgMax/Squeeze", { InferenceEngine::Precision::I32, {1, 513, 513}, InferenceEngine::Layout::CHW} },
                    { "segmentation_map", { InferenceEngine::Precision::I32, {1, 1, 512, 512}, InferenceEngine::Layout::NCHW} },
                    { "softmax", { InferenceEngine::Precision::FP32, {1, 150, 320, 320}, InferenceEngine::Layout::NCHW} },
                    { "L0317_ReWeight_SoftMax",  { InferenceEngine::Precision::FP32, {1, 4, 512, 896}, InferenceEngine::Layout::NCHW } } } },
        }
    );

    return { "segmentation", {inputPattern, outputPattern} };
}

template<class OutputsDataMap>
void SegmentationModel::getBlobDims(const OutputsDataMap& outputInfo) {
    const InferenceEngine::SizeVector outputDims = outputInfo.find(outputsNames[0])->second->getTensorDesc().getDims();
    auto nDims = outputDims.size();
    outChannels = nDims > 3 ? (int)(outputDims[1]) : 0;
    outHeight = (int)(outputDims[nDims - 2]);
    outWidth = (int)(outputDims[nDims - 1]);
}

void SegmentationModel::checkCompiledNetworkInputsOutputs() {
    ImageModel::checkCompiledNetworkInputsOutputs();
    getBlobDims(execNetwork.GetOutputsInfo());
}

void SegmentationModel::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output -------------------------------------------------
    ImageModel::prepareInputsOutputs(cnnNetwork);
    getBlobDims(cnnNetwork.getOutputsInfo());

}

std::shared_ptr<InternalModelData> SegmentationModel::preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) {
    auto imgData = inputData.asRef<ImageInputData>();
    auto& img = imgData.inputImage;

    std::shared_ptr<InternalModelData> resPtr = nullptr;

    if (useAutoResize) {
        /* Just set input blob containing read image. Resize and layout conversionx will be done automatically */
        request->SetBlob(inputsNames[0], wrapMat2Blob(img));
        /* IE::Blob::Ptr from wrapMat2Blob() doesn't own data. Save the image to avoid deallocation before inference */
        resPtr = std::make_shared<InternalImageMatModelData>(img);
    }
    else {
        /* Resize and copy data from the image to the input blob */
        InferenceEngine::Blob::Ptr frameBlob = request->GetBlob(inputsNames[0]);
        matU8ToBlob<uint8_t>(img, frameBlob);
        resPtr = std::make_shared<InternalImageModelData>(img.cols, img.rows);
    }

    return resPtr;
}

std::unique_ptr<ResultBase> SegmentationModel::postprocess(InferenceResult& infResult) {
    SegmentationResult* result = new SegmentationResult(infResult.frameId, infResult.metaData);

    const auto& inputImgSize = infResult.internalModelData->asRef<InternalImageModelData>();

    InferenceEngine::MemoryBlob::Ptr blobPtr = infResult.getFirstOutputBlob();

    void* pData = blobPtr->rmap().as<void*>();

    result->mask = cv::Mat(outHeight, outWidth, CV_8UC1);

    if (outChannels == 1 && blobPtr->getTensorDesc().getPrecision() == InferenceEngine::Precision::I32) {
        cv::Mat predictions(outHeight, outWidth, CV_32SC1, pData);
        predictions.convertTo(result->mask, CV_8UC1);
    }
    else if (blobPtr->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32) {
        float* ptr = reinterpret_cast<float*>(pData);
        for (int rowId = 0; rowId < outHeight; ++rowId) {
            for (int colId = 0; colId < outWidth; ++colId) {
                int classId = 0;
                float maxProb = -1.0f;
                for (int chId = 0; chId < outChannels; ++chId) {
                    float prob = ptr[chId * outHeight * outWidth + rowId * outWidth + colId];
                    if (prob > maxProb) {
                        classId = chId;
                        maxProb = prob;
                    }
                } // nChannels

                result->mask.at<uint8_t>(rowId, colId) = classId;
            } // width
        } // height
    }

    cv::resize(result->mask, result->mask,
        cv::Size(inputImgSize.inputImgWidth, inputImgSize.inputImgHeight),
        0, 0, cv::INTER_NEAREST);

    return std::unique_ptr<ResultBase>(result);
}
