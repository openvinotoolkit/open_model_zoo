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
    const TensorDesc& inputDesc = input->getTensorDesc();
    input->setPrecision(Precision::U8);

    if (inputDesc.getDims()[1] != 3) {
         throw std::logic_error("Expected 3-channel input");
     }

    if (useAutoResize) {
        input->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
        input->getInputData()->setLayout(Layout::NHWC);
    }
    else {
        input->getInputData()->setLayout(Layout::NCHW);
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

    for (auto& output : outputInfo) {
        const TensorDesc& outputDesc = output.second->getTensorDesc();
        maxProposalsCount = outputDesc.getDims()[1];
        objectSize.push_back(outputDesc.getDims()[2]);
        output.second->setPrecision(InferenceEngine::Precision::FP32);
        output.second->setLayout(InferenceEngine::Layout::CHW);
        outputsNames.push_back(output.first);
    }

}

void calculateAnchors(std::vector<ModelFaceBoxes::Anchor>* anchors, const std::vector<double>& vx, const std::vector<double>& vy, int imgWidth, int imgHeight, int minSize,  int step) {
    double skx = static_cast<double>(minSize) / imgWidth;
    double sky = static_cast<double>(minSize) / imgHeight;
    skx = std::min(std::max(skx, 0.), 1.);
    sky = std::min(std::max(sky, 0.), 1.);

    std::vector<double> dense_cx(vx.size()), dense_cy(vy.size);

    for (auto x : vx) {
        dense_cx.push_back(x * step / imgWidth);
    }

    for (auto y : vy) {
        dense_cy.push_back(y * step / imgHeight);
    }

    for (auto cy : dense_cy) {
        for (auto cx : dense_cx) {
            cx = std::min(std::max(cx, 0.), 1.);
            cy = std::min(std::max(cy, 0.), 1.);
            anchors->push_back({ cx, cy, skx, sky });
        }
    }

}

void calculateAnchorsZeroLevel(std::vector<ModelFaceBoxes::Anchor>* anchors, int fx, int fy, int imgWidth, int imgHeight, const std::vector<int>& minSizes, int step) {
    for (auto s : minSizes) {
        std::vector<double> vx, vy;
        if (s == 32) {
            vx.push_back(fx);
            vx.push_back(fx + 0.25);
            vx.push_back(fx + 0.5);
            vx.push_back(fx + 0.75);

            vy.push_back(fy);
            vy.push_back(fy + 0.25);
            vy.push_back(fy + 0.5);
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
        calculateAnchors(anchors, vx, vy, imgWidth, imgHeight, s, step);
    }
}

std::vector<ModelFaceBoxes::Anchor> ModelFaceBoxes::priorBoxes(std::vector<std::pair<int, int>> featureMaps, int imgWidth, int imgHeight) {
    std::vector<ModelFaceBoxes::Anchor> anchors(maxProposalsCount);

    for (int k = 0; k < featureMaps.size(); ++k) {
        std::vector<double> a;
        for (int i = 0; i < featureMaps[k].first; ++i) {
            for (int j = 0; j < featureMaps[k].second; ++j) {
                if (k == 0) {
                    calculateAnchorsZeroLevel(&anchors, j, i, imgWidth,  imgHeight,  minSizes[k], steps[k]);;
                }
                else {
                    calculateAnchors(&anchors, { j + 0.5 }, { i + 0.5 }, imgWidth, imgHeight, minSizes[k][0], steps[k]);
                }
            }
        }
    }

    for (auto& anc : anchors) {
        //anc = std::min(std::max(anc, 0.), 1.);
        //anc.cy = std::min(std::max(anc.cx, 0.), 1.);
        //anc.skx = std::min(std::max(anc.cx, 0.), 1.);
        //anc.sky = std::min(std::max(anc.cx, 0.), 1.);
    }

    return anchors;
}

std::vector<int> nms(const std::vector<ModelFaceBoxes::Anchor>& boxes, const std::vector<double>& scores, double thresh) {

    std::vector<double> areas(boxes.size());

    for (int i = 0; i < boxes.size(); ++i) {
        areas[i] = (boxes[i].right - boxes[i].left) * (boxes[i].bottom - boxes[i].top);
    }
    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&scores](int o1, int o2) { return scores[o1] > scores[o2]; });

    int ordersNum = scores.size();

    std::vector<int> keep;
    bool shouldContinue = true;
    for (int i = 0; shouldContinue && i < ordersNum; ++i) {
        auto idx1 = order[i];
        if (idx1 >= 0) {
            keep.push_back(idx1);
            shouldContinue = false;
            for (int j = i + 1; j < ordersNum; ++j) {
                auto idx2 = order[j];
                if (idx2 >= 0) {
                    shouldContinue = true;
                    double overlappingWidth = fmin(boxes[idx1].right, boxes[idx2].right) - fmax(boxes[idx1].left, boxes[idx2].left);
                    double overlappingHeight = fmin(boxes[idx1].bottom, boxes[idx2].bottom) - fmax(boxes[idx1].top, boxes[idx2].top);
                    auto intersection = overlappingWidth > 0 && overlappingHeight > 0 ? overlappingWidth * overlappingHeight : 0;
                    auto overlap = intersection / (areas[idx1] + areas[idx2] - intersection);

                    if (overlap >= thresh) {
                        order[j] = -1;
                    }
                }
            }
        }
    }
    return keep;
}

std::unique_ptr<ResultBase> ModelFaceBoxes::postprocess(InferenceResult& infResult) {
    auto start = std::chrono::high_resolution_clock::now();
    //size imgSize{ infResult.internalModelData->asRef<InternalImageModelData>().inputImgWidth,infResult.internalModelData->asRef<InternalImageModelData>().inputImgHeight };
   //auto imgWidth = ; // wrong
    //auto imgHeight = ; // wrong 

    std::vector<std::pair<int, int>> featureMaps(steps.size());

    for (auto s : steps) {
        featureMaps.push_back({ netInputHeight / s, netInputWidth / s });
    }

    std::vector<Anchor> priorData = priorBoxes(featureMaps, netInputWidth, netInputHeight);

    auto bboxesInfRes = infResult.outputsData[outputsNames[0]]; //0:21824, each 4 el
    auto desc = bboxesInfRes->getTensorDesc();
    auto bbSize = desc.getDims();
    //auto landmarkPredLen = sz[1] / anchorNum;
    auto blockWidth = bbSize[2];
    LockedMemory<const void> bboxesOutputMapped = bboxesInfRes->rmap();
    std::vector<double> bboxesData(bbSize[1] * bbSize[2]);
    const float *bboxesPtr = bboxesOutputMapped.as<float*>();
    size_t offset = blockWidth / 2;
    for (size_t i = 0; i < bbSize[1] * bbSize[2]; ++i) {
        if (i % blockWidth < offset) {
            bboxesData[i] = bboxesPtr[i] * variance[0] * priorData[i + offset] + priorData[i];
        }
        else {
            bboxesData[i] = exp(bboxesPtr[i] * variance[1]) * priorData[i];
        }

    }

    auto scoresInfRes = infResult.outputsData[outputsNames[1]]; //0:21824, each 2 el
    desc = scoresInfRes->getTensorDesc();
    auto scSize = desc.getDims();
    auto scWidth = scSize[2];
    LockedMemory<const void> scoresOutputMapped = scoresInfRes->rmap();
    std::vector<double> scores(scSize[1] * scSize[2]);
    const float *scoresPtr = scoresOutputMapped.as<float*>();
    std::vector<int> indices;
    std::vector<double> filteredScores;
    for (size_t i = 1; i < scSize[1] * scSize[2]; i = i + 2) {
        if (scoresPtr[i] > confidenceThreshold) {
            indices.push_back(i / 2);
            filteredScores.push_back(scoresPtr[i]);
        }
    }

    std::vector<Anchor> filteredBBoxes;
    auto imgWidth = infResult.internalModelData->asRef<InternalImageModelData>().inputImgWidth;
    auto imgHeight = infResult.internalModelData->asRef<InternalImageModelData>().inputImgHeight;
    // make anchors
    for (auto i : indices) {
        Anchor a;
        a.left = (bboxesData[4 * i] - 0.5*bboxesData[4 * i + 2]) * imgWidth;
        a.right = (bboxesData[4 * i] + 0.5*bboxesData[4 * i + 2]) * imgWidth;
        a.top = (bboxesData[4 * i + 1] - 0.5*bboxesData[4 * i + 3]) * imgHeight;
        a.bottom = (bboxesData[4 * i + 1]  + 0.5*bboxesData[4 * i + 3]) * imgHeight;
        filteredBBoxes.push_back(a);
    }

    std::vector<int> keep = nms(filteredBBoxes, filteredScores, boxIOUThreshold);

    DetectionResult* result = new DetectionResult;
    *static_cast<ResultBase*>(result) = static_cast<ResultBase&>(infResult);
    double scaleX = ((double)netInputWidth) / imgWidth;
    double scaleY = ((double)netInputHeight) / imgHeight;

    result->objects.reserve(keep.size());
    for (auto i : keep) {
        DetectedObject desc;
        desc.confidence = static_cast<float>(filteredScores[i]);
        desc.x = static_cast<float>(filteredBBoxes[i].left);
        desc.y = static_cast<float>(filteredBBoxes[i].top);
        desc.width = static_cast<float>(filteredBBoxes[i].getWidth());
        desc.height = static_cast<float>(filteredBBoxes[i].getHeight());
 
        desc.labelID =  0;
        desc.label = labels[0];
        result->objects.push_back(desc);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    //std::cout << "p " << time_span.count() << "ms" << std::endl;
    return std::unique_ptr<ResultBase>(result);;
}
