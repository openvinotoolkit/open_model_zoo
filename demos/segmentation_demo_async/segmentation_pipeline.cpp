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

#include "segmentation_pipeline.h"
#include <samples/args_helper.hpp>

using namespace InferenceEngine;

SegmentationPipeline::SegmentationPipeline()
    :distr(0, 255) {
    colors.resize(arraySize(CITYSCAPES_COLORS));
    for (std::size_t i = 0; i < colors.size(); ++i)
        colors[i] = { CITYSCAPES_COLORS[i].blue(), CITYSCAPES_COLORS[i].green(), CITYSCAPES_COLORS[i].red() };
}


SegmentationPipeline::~SegmentationPipeline(){
}

void SegmentationPipeline::PrepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork){
    // --------------------------- Configure input & output ---------------------------------------------
    // --------------------------- Prepare input blobs -----------------------------------------------------
    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    if (inputShapes.size() != 1)
        throw std::runtime_error("Demo supports topologies only with 1 input");
    imageInputName = inputShapes.begin()->first;
    SizeVector& inSizeVector = inputShapes.begin()->second;
    if (inSizeVector.size() != 4 || inSizeVector[1] != 3)
        throw std::runtime_error("3-channel 4-dimensional model's input is expected");
    inSizeVector[0] = 1;  // set batch size to 1
    cnnNetwork.reshape(inputShapes);

    InputInfo& inputInfo = *cnnNetwork.getInputsInfo().begin()->second;
    inputInfo.getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
    inputInfo.setLayout(Layout::NHWC);
    inputInfo.setPrecision(Precision::U8);

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
        outHeight = (int)outSizeVector[1];
        outWidth = (int)outSizeVector[2];
        break;
    case 4:
        outChannels = (int)outSizeVector[1];
        outHeight = (int)outSizeVector[2];
        outWidth = (int)outSizeVector[3];
        break;
    default:
        throw std::runtime_error("Unexpected output blob shape. Only 4D and 3D output blobs are"
            "supported.");
    }
}

SegmentationPipeline::SegmentationResult SegmentationPipeline::getSegmentationResult(){
    auto reqResult = PipelineBase::getInferenceResult();
    if (reqResult.IsEmpty()){
        return SegmentationResult();
    }

    LockedMemory<const void> outMapped = reqResult.getFirstOutputBlob()->rmap();
    const float * const predictions = outMapped.as<float*>();

    cv::Mat maskImg(outHeight, outWidth, CV_8UC3);
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

            maskImg.at<cv::Vec3b>(rowId, colId) = class2Color(classId);
        }
    }

    return SegmentationResult(reqResult.frameId,maskImg );
}

const cv::Vec3b& SegmentationPipeline::class2Color(int classId)
{
    while (classId >= (int)colors.size()) {
        cv::Vec3b color(distr(rng), distr(rng), distr(rng));
        colors.push_back(color);
    }
    return colors[classId];
}
