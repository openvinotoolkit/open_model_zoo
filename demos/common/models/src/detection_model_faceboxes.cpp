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

#include "models/detection_model_faceboxes.h"
#include <samples/slog.hpp>
#include <samples/common.hpp>
#include <ngraph/ngraph.hpp>

using namespace InferenceEngine;

ModelFaceBoxes::ModelFaceBoxes(const std::string& modelFileName,
    float confidenceThreshold, bool useAutoResize, float boxIOUThreshold)
    : DetectionModel(modelFileName, confidenceThreshold, useAutoResize, {"Face"}),
    boxIOUThreshold(boxIOUThreshold), variance({0.1, 0.2}), steps({32, 64, 128}), keepTopK(750),
    ms({{32, 64, 128}, 256, 512}) {
}

void ModelFaceBoxes::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
// --------------------------- Configure input & output -------------------------------------------------
// --------------------------- Prepare input blobs ------------------------------------------------------
    slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error("This demo accepts networks that have only one input");
    }
    InputInfo::Ptr& input = inputInfo.begin()->second;
    std::string imageInputName = inputInfo.begin()->first;
    inputsNames.push_back(imageInputName);
    auto ch = cnnNetwork.getInputShapes()[imageInputName][1];
     if (ch != 3) {
         throw std::logic_error("Expected 3-channel input");
     }
    input->setPrecision(Precision::U8);
    if (useAutoResize) {
        input->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
        input->getInputData()->setLayout(Layout::NHWC);
    }
    else {
        input->getInputData()->setLayout(Layout::NCHW);
    }

    //--- Reading image input parameters
    imageInputName = inputInfo.begin()->first;
    const TensorDesc& inputDesc = inputInfo.begin()->second->getTensorDesc();
    netInputHeight = getTensorHeight(inputDesc);
    netInputWidth = getTensorWidth(inputDesc);

// --------------------------- Prepare output blobs -----------------------------------------------------
    slog::info << "Checking that the outputs are as the demo expects" << slog::endl;

    InferenceEngine::OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
    if (outputInfo.size() != 2) {
        throw std::logic_error("This demo expect networks that have 2 outputs blobs");
    }
    for (auto& output : outputInfo) {
        output.second->setPrecision(InferenceEngine::Precision::FP32);
        //output.second->setLayout(InferenceEngine::Layout::NCHW);
        outputsNames.push_back(output.first);
    }

}
std::vector<double> priorBoxes(std::vector<std::pair<int, int>> featureMaps, int imgWidth, int imgHeight) {
    for (int i = 0; i < featureMaps.size(); ++i) {

    }
}

std::unique_ptr<ResultBase> ModelFaceBoxes::postprocess(InferenceResult& infResult) {
    auto bboxes = infResult.outputsData[outputsNames[0]];
    auto scores = infResult.outputsData[outputsNames[1]];
    std::vector<std::pair<int, int>> featureMaps;
    for (auto s : steps) {
        featureMaps.push_back({ netInputHeight / s, netInputWidth / s });
    }

    return std::unique_ptr<ResultBase>();
}
