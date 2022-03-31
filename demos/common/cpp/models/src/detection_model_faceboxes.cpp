/*
// Copyright (C) 2020-2022 Intel Corporation
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

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

#include <openvino/openvino.hpp>

#include <utils/common.hpp>
#include <utils/nms.hpp>
#include <utils/ocv_common.hpp>

#include "models/internal_model_data.h"
#include "models/results.h"

ModelFaceBoxes::ModelFaceBoxes(const std::string& modelFileName,
                               float confidenceThreshold,
                               bool useAutoResize,
                               float boxIOUThreshold,
                               const std::string& layout)
    : DetectionModel(modelFileName, confidenceThreshold, useAutoResize, {"Face"}, layout),
      maxProposalsCount(0),
      boxIOUThreshold(boxIOUThreshold),
      variance({0.1f, 0.2f}),
      steps({32, 64, 128}),
      minSizes({{32, 64, 128}, {256}, {512}}) {}

void ModelFaceBoxes::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input  ------------------------------------------------------
    if (model->inputs().size() != 1) {
        throw std::logic_error("FaceBoxes model wrapper expects models that have only 1 input");
    }

    const ov::Shape& inputShape = model->input().get_shape();
    const ov::Layout& inputLayout = getInputLayout(model->input());

    if (inputShape[ov::layout::channels_idx(inputLayout)] != 3) {
        throw std::logic_error("Expected 3-channel input");
    }

    ov::preprocess::PrePostProcessor ppp(model);
    inputTransform.setPrecision(ppp, model->input().get_any_name());
    ppp.input().tensor().set_layout({"NHWC"});

    if (useAutoResize) {
        ppp.input().tensor().set_spatial_dynamic_shape();

        ppp.input()
            .preprocess()
            .convert_element_type(ov::element::f32)
            .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
    }

    ppp.input().model().set_layout(inputLayout);

    // --------------------------- Reading image input parameters -------------------------------------------
    inputsNames.push_back(model->input().get_any_name());
    netInputWidth = inputShape[ov::layout::width_idx(inputLayout)];
    netInputHeight = inputShape[ov::layout::height_idx(inputLayout)];

    // --------------------------- Prepare output  -----------------------------------------------------
    if (model->outputs().size() != 2) {
        throw std::logic_error("FaceBoxes model wrapper expects models that have 2 outputs");
    }

    const ov::Layout outputLayout{"CHW"};
    maxProposalsCount = model->outputs().front().get_shape()[ov::layout::height_idx(outputLayout)];
    for (const auto& output : model->outputs()) {
        const auto outTensorName = output.get_any_name();
        outputsNames.push_back(outTensorName);
        ppp.output(outTensorName).tensor().set_element_type(ov::element::f32).set_layout(outputLayout);
    }
    std::sort(outputsNames.begin(), outputsNames.end());
    model = ppp.build();

    // --------------------------- Calculating anchors ----------------------------------------------------
    std::vector<std::pair<size_t, size_t>> featureMaps;
    for (auto s : steps) {
        featureMaps.push_back({netInputHeight / s, netInputWidth / s});
    }

    priorBoxes(featureMaps);
}

void calculateAnchors(std::vector<ModelFaceBoxes::Anchor>& anchors,
                      const std::vector<float>& vx,
                      const std::vector<float>& vy,
                      const int minSize,
                      const int step) {
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
            anchors.push_back(
                {cx - 0.5f * skx, cy - 0.5f * sky, cx + 0.5f * skx, cy + 0.5f * sky});  // left top right bottom
        }
    }
}

void calculateAnchorsZeroLevel(std::vector<ModelFaceBoxes::Anchor>& anchors,
                               const int fx,
                               const int fy,
                               const std::vector<int>& minSizes,
                               const int step) {
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
        } else if (s == 64) {
            vx.push_back(static_cast<float>(fx));
            vx.push_back(fx + 0.5f);

            vy.push_back(static_cast<float>(fy));
            vy.push_back(fy + 0.5f);
        } else {
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
                    calculateAnchorsZeroLevel(anchors, j, i, minSizes[k], steps[k]);
                } else {
                    calculateAnchors(anchors, {j + 0.5f}, {i + 0.5f}, minSizes[k][0], steps[k]);
                }
            }
        }
    }
}

std::pair<std::vector<size_t>, std::vector<float>> filterScores(const ov::Tensor& scoresTensor,
                                                                const float confidenceThreshold) {
    auto shape = scoresTensor.get_shape();
    const float* scoresPtr = scoresTensor.data<float>();

    std::vector<size_t> indices;
    std::vector<float> scores;
    scores.reserve(ModelFaceBoxes::INIT_VECTOR_SIZE);
    indices.reserve(ModelFaceBoxes::INIT_VECTOR_SIZE);
    for (size_t i = 1; i < shape[1] * shape[2]; i = i + 2) {
        if (scoresPtr[i] > confidenceThreshold) {
            indices.push_back(i / 2);
            scores.push_back(scoresPtr[i]);
        }
    }

    return {indices, scores};
}

std::vector<ModelFaceBoxes::Anchor> filterBoxes(const ov::Tensor& boxesTensor,
                                                const std::vector<ModelFaceBoxes::Anchor>& anchors,
                                                const std::vector<size_t>& validIndices,
                                                const std::vector<float>& variance) {
    auto shape = boxesTensor.get_shape();
    const float* boxesPtr = boxesTensor.data<float>();

    std::vector<ModelFaceBoxes::Anchor> boxes;
    boxes.reserve(ModelFaceBoxes::INIT_VECTOR_SIZE);
    for (auto i : validIndices) {
        auto objStart = shape[2] * i;

        auto dx = boxesPtr[objStart];
        auto dy = boxesPtr[objStart + 1];
        auto dw = boxesPtr[objStart + 2];
        auto dh = boxesPtr[objStart + 3];

        auto predCtrX = dx * variance[0] * anchors[i].getWidth() + anchors[i].getXCenter();
        auto predCtrY = dy * variance[0] * anchors[i].getHeight() + anchors[i].getYCenter();
        auto predW = exp(dw * variance[1]) * anchors[i].getWidth();
        auto predH = exp(dh * variance[1]) * anchors[i].getHeight();

        boxes.push_back({static_cast<float>(predCtrX - 0.5f * predW),
                         static_cast<float>(predCtrY - 0.5f * predH),
                         static_cast<float>(predCtrX + 0.5f * predW),
                         static_cast<float>(predCtrY + 0.5f * predH)});
    }

    return boxes;
}

std::unique_ptr<ResultBase> ModelFaceBoxes::postprocess(InferenceResult& infResult) {
    // Filter scores and get valid indices for bounding boxes
    const auto scoresTensor = infResult.outputsData[outputsNames[1]];
    const auto scores = filterScores(scoresTensor, confidenceThreshold);

    // Filter bounding boxes on indices
    auto boxesTensor = infResult.outputsData[outputsNames[0]];
    std::vector<Anchor> boxes = filterBoxes(boxesTensor, anchors, scores.first, variance);

    // Apply Non-maximum Suppression
    const std::vector<int> keep = nms(boxes, scores.second, boxIOUThreshold);

    // Create detection result objects
    DetectionResult* result = new DetectionResult(infResult.frameId, infResult.metaData);
    const auto imgWidth = infResult.internalModelData->asRef<InternalImageModelData>().inputImgWidth;
    const auto imgHeight = infResult.internalModelData->asRef<InternalImageModelData>().inputImgHeight;
    const float scaleX = static_cast<float>(netInputWidth) / imgWidth;
    const float scaleY = static_cast<float>(netInputHeight) / imgHeight;

    result->objects.reserve(keep.size());
    for (auto i : keep) {
        DetectedObject desc;
        desc.confidence = scores.second[i];
        desc.x = clamp(boxes[i].left / scaleX, 0.f, static_cast<float>(imgWidth));
        desc.y = clamp(boxes[i].top / scaleY, 0.f, static_cast<float>(imgHeight));
        desc.width = clamp(boxes[i].getWidth() / scaleX, 0.f, static_cast<float>(imgWidth));
        desc.height = clamp(boxes[i].getHeight() / scaleY, 0.f, static_cast<float>(imgHeight));
        desc.labelID = 0;
        desc.label = labels[0];

        result->objects.push_back(desc);
    }

    return std::unique_ptr<ResultBase>(result);
}
