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

#include "models/segmentation_model.h"
#include "utils/ocv_common.hpp"

using namespace InferenceEngine;

SegmentationModel::SegmentationModel(const std::string& modelFileName, bool useAutoResize) :
    ModelBase(modelFileName),
    useAutoResize(useAutoResize) {
}

void SegmentationModel::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output ---------------------------------------------
    // --------------------------- Prepare input blobs -----------------------------------------------------
    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    if (inputShapes.size() != 1)
        throw std::runtime_error("Demo supports topologies only with 1 input");
    inputsNames.push_back(inputShapes.begin()->first);
    SizeVector& inSizeVector = inputShapes.begin()->second;
    if (inSizeVector.size() != 4 || inSizeVector[1] != 3)
        throw std::runtime_error("3-channel 4-dimensional model's input is expected");

    InputInfo& inputInfo = *cnnNetwork.getInputsInfo().begin()->second;
    inputInfo.setPrecision(Precision::U8);
    if (useAutoResize) {
        inputInfo.getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
        inputInfo.setLayout(Layout::NHWC);
    }
    else {
        inputInfo.setLayout(Layout::NCHW);
    }
    // --------------------------- Prepare output blobs -----------------------------------------------------
    const OutputsDataMap& outputsDataMap = cnnNetwork.getOutputsInfo();
    if (outputsDataMap.size() != 1) throw std::runtime_error("Demo supports topologies only with 1 output");

    outputsNames.push_back(outputsDataMap.begin()->first);
    Data& data = *outputsDataMap.begin()->second;
    // if the model performs ArgMax, its output type can be I32 but for models that return heatmaps for each
    // class the output is usually FP32. Reset the precision to avoid handling different types with switch in
    // postprocessing
    data.setPrecision(Precision::FP32);
    const SizeVector& outSizeVector = data.getTensorDesc().getDims();
    switch (outSizeVector.size()) {
    case 3:
        outChannels = 0;
        outHeight = (int)(outSizeVector[1]);
        outWidth = (int)(outSizeVector[2]);
        break;
    case 4:
        outChannels = (int)(outSizeVector[1]);
        outHeight = (int)(outSizeVector[2]);
        outWidth = (int)(outSizeVector[3]);
        break;
    default:
        throw std::runtime_error("Unexpected output blob shape. Only 4D and 3D output blobs are"
            "supported.");
    }
}

std::shared_ptr<InternalModelData> SegmentationModel::preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) {
    auto imgData = inputData.asRef<ImageInputData>();
    auto& img = imgData.inputImage;

    if (useAutoResize) {
        /* Just set input blob containing read image. Resize and layout conversionx will be done automatically */
        request->SetBlob(inputsNames[0], wrapMat2Blob(img));
        /* IE::Blob::Ptr from wrapMat2Blob() doesn't own data. Save the image to avoid deallocation before inference */
        return std::make_shared<InternalImageMatModelData>(img);
    }
    /* Resize and copy data from the image to the input blob */
    Blob::Ptr frameBlob = request->GetBlob(inputsNames[0]);
    matU8ToBlob<uint8_t>(img, frameBlob);
    return std::make_shared<InternalImageModelData>(img.cols, img.rows);
}

std::unique_ptr<ResultBase> SegmentationModel::postprocess(InferenceResult& infResult) {
    SegmentationResult* result = new SegmentationResult;
    *static_cast<ResultBase*>(result) = static_cast<ResultBase&>(infResult);

    const auto& inputImgSize = infResult.internalModelData->asRef<InternalImageModelData>();

    LockedMemory<const void> outMapped = infResult.getFirstOutputBlob()->rmap();
    const float * const predictions = outMapped.as<float*>();

    result->mask = cv::Mat(outHeight, outWidth, CV_8UC1);
    for (int rowId = 0; rowId < outHeight; ++rowId) {
        for (int colId = 0; colId < outWidth; ++colId) {
            std::size_t classId = 0;
            if (outChannels < 2) {  // assume the output is already ArgMax'ed
                classId = static_cast<std::size_t>(predictions[rowId * outWidth + colId]);
            }
            else {
                float maxProb = -1.0f;
                for (int chId = 0; chId < outChannels; ++chId) {
                    float prob = predictions[chId * outHeight * outWidth + rowId * outWidth + colId];
                    if (prob > maxProb) {
                        classId = chId;
                        maxProb = prob;
                    }
                }
            }

            result->mask.at<uint8_t>(rowId, colId) = classId;
        }
    }
    cv::resize(result->mask, result->mask, cv::Size(inputImgSize.inputImgWidth, inputImgSize.inputImgHeight),0,0,cv::INTER_NEAREST);

    return std::unique_ptr<ResultBase>(result);
}
