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

#include <algorithm>
#include "models/detection_model_faceboxes.h"
#include <samples/slog.hpp>
#include <samples/common.hpp>
#include <ngraph/ngraph.hpp>

using namespace InferenceEngine;
using size = std::pair<int, int>;

ModelFaceBoxes::ModelFaceBoxes(const std::string& modelFileName,
    float confidenceThreshold, bool useAutoResize, float boxIOUThreshold)
    : DetectionModel(modelFileName, confidenceThreshold, useAutoResize, {"Face"}),
    boxIOUThreshold(boxIOUThreshold), variance({0.1, 0.2}), steps({32, 64, 128}), keepTopK(750),
    minSizes({ {32, 64, 128}, {256}, {512} }) {
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

std::vector<ModelFaceBoxes::Anchor> calculateAnchors(std::vector<double> vx, std::vector<double> vy, int minSize,
    int imgWidth, int imgHeight, int step) {
    std::vector<ModelFaceBoxes::Anchor> anchors;
    double skx = minSize / imgWidth;
    double sky = minSize / imgHeight;
    std::vector<double> dense_cx, dense_cy;

    for (auto x : vx) {
        dense_cx.push_back(x * step / imgWidth);
    }

    for (auto y : vy) {
        dense_cx.push_back(y * step / imgHeight);
    }

    for (auto cy : dense_cy) {
        for (auto cx : dense_cx) {
            anchors.push_back({ cx, cy, skx, sky });
        }
    }

    return anchors;
}

std::vector<ModelFaceBoxes::Anchor> calculateAnchorsZeroLevel(int fx, int fy,  std::vector<int> minSizes, int imgWidth, int imgHeight, int step) {
    std::vector<ModelFaceBoxes::Anchor> anchors;
    std::vector<double> vx, vy;
    for (auto s : minSizes) {
        if (s == 32) {
            vx.push_back(fx);
            vx.push_back(fx + 0.25);
            vx.push_back(fx + 0.75);

            vy.push_back(fy);
            vy.push_back(fy + 0.25);
            vy.push_back(fy + 0.75);
        }
        else if (s == 64) {
            vx.push_back(fx);
            vx.push_back(fx + 0.5);

            vy.push_back(fy);
            vy.push_back(fy + 0.5);
        }
        else {
            vx.push_back(fx + 0.5);
            vy.push_back(fy + 0.5);
        }
        auto a = calculateAnchors(vx, vy, s, imgWidth, imgHeight, step);
        anchors.insert(anchors.end(), a.begin(), a.end());
    }
    return anchors;
}

std::vector<ModelFaceBoxes::Anchor> ModelFaceBoxes::priorBoxes(std::vector<std::pair<int, int>> featureMaps, int imgWidth, int imgHeight) {
    std::vector<Anchor> anchors;

    for (int k = 1; k < featureMaps.size(); ++k) {
        std::vector<Anchor> a;
        for (int i = 0; i < featureMaps[i].first; ++i) {
            for (int j = 0; j < featureMaps[j].second; ++j) {
                if (k == 0) {
                    a = calculateAnchorsZeroLevel(j, i, minSizes[k], imgWidth, imgHeight, steps[k]);;
                }
                else {
                    a = calculateAnchors({ j + 0.5 }, { i + 0.5 }, minSizes[k][0], imgWidth, imgHeight, steps[k]);
                }

                anchors.insert(anchors.end(), a.begin(), a.end());
            }
        }
    }

    for (auto& anc : anchors) {
        anc.cx = std::min(std::max(anc.cx, 0.), 1.);
        anc.cy = std::min(std::max(anc.cx, 0.), 1.);
        anc.skx = std::min(std::max(anc.cx, 0.), 1.);
        anc.sky = std::min(std::max(anc.cx, 0.), 1.);
    }
}

std::unique_ptr<ResultBase> ModelFaceBoxes::postprocess(InferenceResult& infResult) {
    //size imgSize{ infResult.internalModelData->asRef<InternalImageModelData>().inputImgWidth,infResult.internalModelData->asRef<InternalImageModelData>().inputImgHeight };
    auto imgWidth = infResult.internalModelData->asRef<InternalImageModelData>().inputImgWidth;
    auto imgHeight = infResult.internalModelData->asRef<InternalImageModelData>().inputImgHeight;
    auto bboxes = infResult.outputsData[outputsNames[0]]; //0:21824, each 4 el
    auto scores = infResult.outputsData[outputsNames[1]]; //0:21824, each 2 el
    std::vector<std::pair<int, int>> featureMaps;

    for (auto s : steps) {
        featureMaps.push_back({ netInputHeight / s, netInputWidth / s });
    }

    std::vector<Anchor> priorData = priorBoxes(featureMaps, imgWidth, imgHeight);

    return std::unique_ptr<ResultBase>();
}
