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
#include <ngraph/ngraph.hpp>
#include <utils/common.hpp>
#include <utils/slog.hpp>
#include "models/detection_model_faceboxes.h"

ModelFaceBoxes::ModelFaceBoxes(const std::string& modelFileName,
    float confidenceThreshold, bool useAutoResize, float boxIOUThreshold)
    : DetectionModel(modelFileName, confidenceThreshold, useAutoResize, {"Face"}),
      maxProposalsCount(0), boxIOUThreshold(boxIOUThreshold), variance({0.1f, 0.2f}),
      steps({32, 64, 128}), minSizes({ {32, 64, 128}, {256}, {512} }) {
}

void ModelFaceBoxes::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input blobs ------------------------------------------------------
    slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
    InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());

    if (inputInfo.size() != 1) {
        throw std::logic_error("This demo accepts networks that have only one input");
    }

    InferenceEngine::InputInfo::Ptr& input = inputInfo.begin()->second;
    const InferenceEngine::TensorDesc& inputDesc = input->getTensorDesc();
    input->setPrecision(InferenceEngine::Precision::U8);

    if (inputDesc.getDims()[1] != 3) {
         throw std::logic_error("Expected 3-channel input");
     }

    if (useAutoResize) {
        input->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
        input->getInputData()->setLayout(InferenceEngine::Layout::NHWC);
    }
    else {
        input->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
    }

    // --------------------------- Reading image input parameters -------------------------------------------
    std::string imageInputName = inputInfo.begin()->first;
    inputsNames.push_back(imageInputName);
    netInputHeight = getTensorHeight(inputDesc);
    netInputWidth = getTensorWidth(inputDesc);

    // --------------------------- Prepare output blobs -----------------------------------------------------
    slog::info << "Checking that the outputs are as the demo expects" << slog::endl;

    InferenceEngine::OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());

    if (outputInfo.size() != 2) {
        throw std::logic_error("This demo expect networks that have 2 outputs blobs");
    }

    const InferenceEngine::TensorDesc& outputDesc = outputInfo.begin()->second->getTensorDesc();
    maxProposalsCount = outputDesc.getDims()[1];

    for (auto& output : outputInfo) {
        output.second->setPrecision(InferenceEngine::Precision::FP32);
        output.second->setLayout(InferenceEngine::Layout::CHW);
        outputsNames.push_back(output.first);
    }

    // --------------------------- Calculating anchors ----------------------------------------------------
    std::vector<std::pair<size_t, size_t>> featureMaps;
    for (auto s : steps) {
        featureMaps.push_back({ netInputHeight / s, netInputWidth / s });
    }

    priorBoxes(featureMaps);

}

void calculateAnchors(std::vector<ModelFaceBoxes::Anchor>& anchors, const std::vector<float>& vx, const std::vector<float>& vy,
    const int minSize, const int step) {
    float skx = static_cast<float>(minSize);
    float sky = static_cast<float>(minSize);

    std::vector<float> dense_cx, dense_cy;

    for (auto x : vx) {
        dense_cx.push_back(x * step);
    }

    for (auto y : vy) {
        dense_cy.push_back(y * step);
    }

    for (auto cy : dense_cy) {
        for (auto cx : dense_cx) {
            anchors.push_back({ cx - 0.5f * skx, cy - 0.5f * sky,
                 cx + 0.5f * skx, cy + 0.5f * sky });  // left top right bottom
        }
    }

}

void calculateAnchorsZeroLevel(std::vector<ModelFaceBoxes::Anchor>& anchors, const int fx, const int fy,
    const std::vector<int>& minSizes, const int step) {
    for (auto s : minSizes) {
        std::vector<float> vx, vy;
        if (s == 32) {
            vx.push_back(static_cast<float>(fx));
            vx.push_back(fx + 0.25f);
            vx.push_back(fx + 0.5f);
            vx.push_back(fx + 0.75f);

            vy.push_back(static_cast<float>(fy));
            vy.push_back(fy + 0.25f);
            vy.push_back(fy + 0.5f);
            vy.push_back(fy + 0.75f);
        }
        else if (s == 64) {
            vx.push_back(static_cast<float>(fx));
            vx.push_back(fx + 0.5f);

            vy.push_back(static_cast<float>(fy));
            vy.push_back(fy + 0.5f);
        }
        else {
            vx.push_back(fx + 0.5f);
            vy.push_back(fy + 0.5f);
        }
        calculateAnchors(anchors, vx, vy, s, step);
    }
}

void ModelFaceBoxes::priorBoxes(const std::vector<std::pair<size_t, size_t>>& featureMaps) {
    anchors.reserve(maxProposalsCount);

    for (size_t k = 0; k < featureMaps.size(); ++k) {
        std::vector<float> a;
        for (size_t i = 0; i < featureMaps[k].first; ++i) {
            for (size_t j = 0; j < featureMaps[k].second; ++j) {
                if (k == 0) {
                    calculateAnchorsZeroLevel(anchors, j, i,  minSizes[k], steps[k]);;
                }
                else {
                    calculateAnchors(anchors, { j + 0.5f }, { i + 0.5f }, minSizes[k][0], steps[k]);
                }
            }
        }
    }
}

std::pair<std::vector<size_t>, std::vector<float>> filterScores(const InferenceEngine::MemoryBlob::Ptr& scoreInfRes, const float confidenceThreshold) {
    auto desc = scoreInfRes->getTensorDesc();
    auto sz = desc.getDims();
    InferenceEngine::LockedMemory<const void> outputMapped = scoreInfRes->rmap();
    const float* scoresPtr = outputMapped.as<float*>();

    std::vector<size_t> indices;
    std::vector<float> scores;
    scores.reserve(ModelFaceBoxes::INIT_VECTOR_SIZE);
    indices.reserve(ModelFaceBoxes::INIT_VECTOR_SIZE);
    for (size_t i = 1; i < sz[1] * sz[2]; i = i + 2) {
        if (scoresPtr[i] > confidenceThreshold) {
            indices.push_back(i / 2);
            scores.push_back(scoresPtr[i]);
        }
    }

    return { indices, scores };
}

std::vector<ModelFaceBoxes::Anchor> filterBBoxes(const InferenceEngine::MemoryBlob::Ptr& bboxesInfRes, const std::vector<ModelFaceBoxes::Anchor>& anchors,
    const std::vector<size_t>& validIndices, const std::vector<float>& variance) {
    InferenceEngine::LockedMemory<const void> bboxesOutputMapped = bboxesInfRes->rmap();
    auto desc = bboxesInfRes->getTensorDesc();
    auto sz = desc.getDims();
    const float *bboxesPtr = bboxesOutputMapped.as<float*>();

    std::vector<ModelFaceBoxes::Anchor> bboxes;
    bboxes.reserve(ModelFaceBoxes::INIT_VECTOR_SIZE);
    for (auto i : validIndices) {
        auto objStart = sz[2] * i;

        auto dx = bboxesPtr[objStart];
        auto dy = bboxesPtr[objStart + 1];
        auto dw = bboxesPtr[objStart + 2];
        auto dh = bboxesPtr[objStart + 3];

        auto predCtrX = dx * variance[0] * anchors[i].getWidth() + anchors[i].getXCenter();
        auto predCtrY = dy * variance[0] * anchors[i].getHeight() + anchors[i].getYCenter();
        auto predW = exp(dw * variance[1]) * anchors[i].getWidth();
        auto predH = exp(dh * variance[1]) * anchors[i].getHeight();

        bboxes.push_back({ static_cast<float>(predCtrX - 0.5f * predW), static_cast<float>(predCtrY - 0.5f * predH),
                                     static_cast<float>(predCtrX + 0.5f * predW), static_cast<float>(predCtrY + 0.5f * predH) });

    }

    return bboxes;
}

std::unique_ptr<ResultBase> ModelFaceBoxes::postprocess(InferenceResult& infResult) {
    // --------------------------- Filter scores and get valid indices for bounding boxes----------------------------------
    const auto scoresInfRes = infResult.outputsData[outputsNames[1]];
    auto scores = filterScores(scoresInfRes, confidenceThreshold);

    // --------------------------- Filter bounding boxes on indices -------------------------------------------------------
    auto bboxesInfRes = infResult.outputsData[outputsNames[0]];
    std::vector<Anchor> bboxes = filterBBoxes(bboxesInfRes, anchors, scores.first, variance);

    // --------------------------- Apply Non-maximum Suppression ----------------------------------------------------------
    std::vector<int> keep = nms(bboxes, scores.second, boxIOUThreshold);

    // --------------------------- Create detection result objects --------------------------------------------------------
    DetectionResult* result = new DetectionResult(infResult.frameId, infResult.metaData);
    auto imgWidth = infResult.internalModelData->asRef<InternalImageModelData>().inputImgWidth;
    auto imgHeight = infResult.internalModelData->asRef<InternalImageModelData>().inputImgHeight;
    float scaleX = static_cast<float>(netInputWidth) / imgWidth;
    float scaleY = static_cast<float>(netInputHeight) / imgHeight;

    result->objects.reserve(keep.size());
    for (auto i : keep) {
        DetectedObject desc;
        desc.confidence = scores.second[i];
        desc.x = bboxes[i].left / scaleX;
        desc.y = bboxes[i].top / scaleY;
        desc.width = bboxes[i].getWidth() / scaleX;
        desc.height = bboxes[i].getHeight() / scaleY;
        desc.labelID =  0;
        desc.label = labels[0];

        result->objects.push_back(desc);
    }

    return std::unique_ptr<ResultBase>(result);
}
