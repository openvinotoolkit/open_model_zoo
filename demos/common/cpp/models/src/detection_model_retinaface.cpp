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

#include <openvino/openvino.hpp>
#include <utils/common.hpp>
#include "models/detection_model_retinaface.h"

ModelRetinaFace::ModelRetinaFace(const std::string& modelFileName, float confidenceThreshold, float boxIOUThreshold,
    const std::string& layout)
    : DetectionModel(modelFileName, confidenceThreshold, {"Face"}, layout),  // Default label is "Face"
    shouldDetectMasks(false), shouldDetectLandmarks(false), boxIOUThreshold(boxIOUThreshold), maskThreshold(0.8f), landmarkStd(1.0f),
    anchorCfg({ {32, { 32, 16 }, 16, { 1 }},
              { 16, { 8, 4 }, 16, { 1 }},
              { 8, { 2, 1 }, 16, { 1 }} }) {
    generateAnchorsFpn();
}

void ModelRetinaFace::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input  ------------------------------------------------------
    if (model->inputs().size() != 1) {
        throw std::logic_error("RetinaFace model wrapper expects models that have only one input");
    }
    const ov::Shape& inputShape = model->input().get_shape();
    ov::Layout inputLayout;
    if (!layouts.empty()) {
        inputLayout = layouts.begin()->second;
    }
    else {
        inputLayout = getLayoutFromShape(inputShape);
    }


    if (inputShape[ov::layout::channels_idx(inputLayout)] != 3) {
        throw std::logic_error("Expected 3-channel input");
    }

    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor().
        set_element_type(ov::element::u8).
        set_spatial_dynamic_shape().
        set_layout({ "NHWC" });

    ppp.input().preprocess().
        convert_element_type(ov::element::f32).
        resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);

    ppp.input().model().set_layout(inputLayout);

    // --------------------------- Reading image input parameters -------------------------------------------
    inputsNames.push_back(model->input().get_any_name());
    netInputWidth = inputShape[ov::layout::width_idx(inputLayout)];
    netInputHeight = inputShape[ov::layout::height_idx(inputLayout)];

    // --------------------------- Prepare output  -----------------------------------------------------

    const ov::OutputVector& outputs = model->outputs();
    if (outputs.size() != 6 && outputs.size() != 9 && outputs.size() != 12) {
        throw std::logic_error("RetinaFace model wrapper expects models that have 6, 9 or 12 outputs");
    }

    ov::Layout outLayout{ "NCHW" };
    std::vector<size_t> outputsSizes[OT_MAX];
    for (const auto& output : model->outputs()) {
        auto outTensorName = output.get_any_name();
        outputsNames.push_back(outTensorName);
        ppp.output(outTensorName).tensor().
            set_element_type(ov::element::f32).
            set_layout(outLayout);

        EOutputType type = OT_MAX;
        if (outTensorName.find("bbox") != std::string::npos) {
            type = OT_BBOX;
        }
        else if (outTensorName.find("cls") != std::string::npos) {
            type = OT_SCORES;
        }
        else if (outTensorName.find("landmark") != std::string::npos) {
            type = OT_LANDMARK;
            shouldDetectLandmarks = true;
        }
        else if (outTensorName.find("type") != std::string::npos) {
            type = OT_MASKSCORES;
            labels.clear();
            labels.push_back("No Mask");
            labels.push_back("Mask");
            shouldDetectMasks = true;
            landmarkStd = 0.2f;
        }
        else {
            continue;
        }

        size_t num = output.get_shape()[ov::layout::height_idx(outLayout)];
        size_t i = 0;
        for (; i < outputsSizes[type].size(); ++i) {
            if (num < outputsSizes[type][i]) {
                break;
            }
        }
        separateOutputsNames[type].insert(separateOutputsNames[type].begin() + i, outTensorName);
        outputsSizes[type].insert(outputsSizes[type].begin() + i, num);

    }
    model = ppp.build();

    for (size_t idx = 0; idx < outputsSizes[OT_BBOX].size(); ++idx) {
        size_t width = outputsSizes[OT_BBOX][idx];
        size_t height = outputsSizes[OT_BBOX][idx];
        auto s = anchorCfg[idx].stride;
        auto anchorNum = anchorsFpn[s].size();

        anchors.push_back(std::vector<ModelRetinaFace::Anchor>(height * width * anchorNum));
        for (size_t iw = 0; iw < width; ++iw) {
            size_t sw = iw * s;
            for (size_t ih = 0; ih < height; ++ih) {
                size_t sh = ih * s;
                for (size_t k = 0; k < anchorNum; ++k) {
                    Anchor& anc = anchors[idx][(ih * width + iw) * anchorNum + k];
                    anc.left = anchorsFpn[s][k].left + sw;
                    anc.top = anchorsFpn[s][k].top + sh;
                    anc.right = anchorsFpn[s][k].right + sw;
                    anc.bottom = anchorsFpn[s][k].bottom + sh;
                }
            }
        }
    }
}

std::vector<ModelRetinaFace::Anchor> ratioEnum(const ModelRetinaFace::Anchor& anchor, const std::vector<int>& ratios) {
    std::vector<ModelRetinaFace::Anchor> retVal;
    auto w = anchor.getWidth();
    auto h = anchor.getHeight();
    auto xCtr = anchor.getXCenter();
    auto yCtr = anchor.getYCenter();

    for (auto ratio : ratios) {
        auto size = w * h;
        auto sizeRatio = static_cast<float>(size) / ratio;
        auto ws = sqrt(sizeRatio);
        auto hs = ws * ratio;
        retVal.push_back({ static_cast<float>(xCtr - 0.5f * (ws - 1.0f)), static_cast<float>(yCtr - 0.5f * (hs - 1.0f)),
            static_cast<float>(xCtr + 0.5f * (ws - 1.0f)), static_cast<float>(yCtr + 0.5f * (hs - 1.0f)) });
    }
    return retVal;
}

std::vector<ModelRetinaFace::Anchor> scaleEnum(const ModelRetinaFace::Anchor& anchor, const std::vector<int>& scales) {
    std::vector<ModelRetinaFace::Anchor> retVal;
    auto w = anchor.getWidth();
    auto h = anchor.getHeight();
    auto xCtr = anchor.getXCenter();
    auto yCtr = anchor.getYCenter();

    for (auto scale : scales) {
        auto ws = w * scale;
        auto hs = h * scale;
        retVal.push_back({ static_cast<float>(xCtr - 0.5f * (ws - 1.0f)),  static_cast<float>(yCtr - 0.5f * (hs - 1.0f)),
            static_cast<float>(xCtr + 0.5f * (ws - 1.0f)),  static_cast<float>(yCtr + 0.5f * (hs - 1.0f)) });
    }
    return retVal;
}

std::vector<ModelRetinaFace::Anchor> generateAnchors(const int baseSize, const std::vector<int>& ratios, const std::vector<int>& scales) {
    ModelRetinaFace::Anchor baseAnchor{ 0.0f, 0.0f, baseSize - 1.0f, baseSize - 1.0f };
    auto ratioAnchors = ratioEnum(baseAnchor, ratios);
    std::vector<ModelRetinaFace::Anchor> retVal;

    for (const auto& ra : ratioAnchors) {
        auto addon = scaleEnum(ra, scales);
        retVal.insert(retVal.end(), addon.begin(), addon.end());
    }
    return retVal;
}

void ModelRetinaFace::generateAnchorsFpn() {
    auto cfg = anchorCfg;
    std::sort(cfg.begin(), cfg.end(), [](const AnchorCfgLine& x, const AnchorCfgLine& y) { return x.stride > y.stride; });

    for (const auto& cfgLine : cfg) {
        anchorsFpn.emplace(cfgLine.stride, generateAnchors(cfgLine.baseSize, cfgLine.ratios, cfgLine.scales));
    }
}


std::vector<size_t> thresholding(const ov::Tensor& scoresTensor, const int anchorNum, const float confidenceThreshold) {
    std::vector<size_t> indices;
    indices.reserve(ModelRetinaFace::INIT_VECTOR_SIZE);
    auto shape = scoresTensor.get_shape();
    size_t restAnchors = shape[1] - anchorNum;
    const float* scoresPtr = scoresTensor.data<float>();

    for (size_t x = anchorNum; x < shape[1]; ++x) {
        for (size_t y = 0; y < shape[2]; ++y) {
            for (size_t z = 0; z < shape[3]; ++z) {
                auto idx = (x * shape[2] + y) * shape[3] + z;
                auto score = scoresPtr[idx];
                if (score >= confidenceThreshold) {
                    indices.push_back((y * shape[3] + z) * restAnchors + (x - anchorNum));
                }
            }
        }
    }

    return indices;
}

void filterScores(std::vector<float>& scores, const std::vector<size_t>& indices, const ov::Tensor& scoresTensor, const int anchorNum) {
    auto shape = scoresTensor.get_shape();
    const float* scoresPtr = scoresTensor.data<float>();
    auto start = shape[2] * shape[3] * anchorNum;

    for (auto i : indices) {
        auto offset = (i % anchorNum) * shape[2] * shape[3] + i / anchorNum;
        scores.push_back(scoresPtr[start + offset]);
    }
}

void filterBBoxes(std::vector<ModelRetinaFace::Anchor>& bboxes, const std::vector<size_t>& indices, const ov::Tensor& bboxesTensor,
    int anchorNum, const std::vector<ModelRetinaFace::Anchor>& anchors) {
    auto shape = bboxesTensor.get_shape();
    const float* bboxesPtr = bboxesTensor.data<float>();
    auto bboxPredLen = shape[1] / anchorNum;
    auto blockWidth = shape[2] * shape[3];


    for (auto i : indices) {
        auto offset = blockWidth * bboxPredLen * (i % anchorNum) + (i / anchorNum);

        auto dx = bboxesPtr[offset];
        auto dy = bboxesPtr[offset + blockWidth];
        auto dw = bboxesPtr[offset + blockWidth * 2];
        auto dh = bboxesPtr[offset + blockWidth * 3];

        auto predCtrX = dx * anchors[i].getWidth() + anchors[i].getXCenter();
        auto predCtrY = dy * anchors[i].getHeight() + anchors[i].getYCenter();
        auto predW = exp(dw) * anchors[i].getWidth();
        auto predH = exp(dh) * anchors[i].getHeight();

        bboxes.push_back({ static_cast<float>(predCtrX - 0.5f * (predW - 1.0f)), static_cast<float>(predCtrY - 0.5f * (predH - 1.0f)),
           static_cast<float>(predCtrX + 0.5f * (predW - 1.0f)), static_cast<float>(predCtrY + 0.5f * (predH - 1.0f)) });
    }
}


void filterLandmarks(std::vector<cv::Point2f>& landmarks, const std::vector<size_t>& indices, const ov::Tensor& landmarksTensor,
        int anchorNum, const std::vector<ModelRetinaFace::Anchor>& anchors, const float landmarkStd) {
    auto shape = landmarksTensor.get_shape();
    const float* landmarksPtr = landmarksTensor.data<float>();
    auto landmarkPredLen = shape[1] / anchorNum;
    auto blockWidth = shape[2] * shape[3];

    for (auto i : indices) {
        for (int j = 0; j < ModelRetinaFace::LANDMARKS_NUM; ++j) {
            auto offset = (i % anchorNum) * landmarkPredLen * shape[2] * shape[3] + i / anchorNum;
            auto deltaX = landmarksPtr[offset + j * 2 * blockWidth] * landmarkStd;
            auto deltaY = landmarksPtr[offset + (j * 2 + 1) * blockWidth] * landmarkStd;
            landmarks.push_back({deltaX * anchors[i].getWidth() + anchors[i].getXCenter(),
               deltaY * anchors[i].getHeight() + anchors[i].getYCenter() });
        }
    }
}

void filterMasksScores(std::vector<float>& masks, const std::vector<size_t>& indices, const ov::Tensor& maskScoresTensor, const int anchorNum) {
    auto shape = maskScoresTensor.get_shape();
    const float* maskScoresPtr = maskScoresTensor.data<float>();
    auto start = shape[2] * shape[3] * anchorNum * 2;

    for (auto i : indices) {
        auto offset = (i % anchorNum) * shape[2] * shape[3] + i / anchorNum;
        masks.push_back(maskScoresPtr[start + offset]);
    }
}

std::unique_ptr<ResultBase> ModelRetinaFace::postprocess(InferenceResult& infResult) {
    std::vector<float> scores;
    scores.reserve(INIT_VECTOR_SIZE);
    std::vector<Anchor> bboxes;
    bboxes.reserve(INIT_VECTOR_SIZE);
    std::vector<cv::Point2f> landmarks;
    std::vector<float> masks;

    if (shouldDetectLandmarks) {
        landmarks.reserve(INIT_VECTOR_SIZE);
    }
    if (shouldDetectMasks) {
        masks.reserve(INIT_VECTOR_SIZE);
    }

    // --------------------------- Gather & Filter output from all levels ----------------------------------------------------------
    for (size_t idx = 0; idx < anchorCfg.size(); ++idx) {
        const auto bboxRaw = infResult.outputsData[separateOutputsNames[OT_BBOX][idx]];
        const auto scoresRaw = infResult.outputsData[separateOutputsNames[OT_SCORES][idx]];
        auto s = anchorCfg[idx].stride;
        auto anchorNum = anchorsFpn[s].size();

        auto validIndices = thresholding(scoresRaw, anchorNum, confidenceThreshold);
        filterScores(scores, validIndices, scoresRaw, anchorNum);
        filterBBoxes(bboxes, validIndices, bboxRaw, anchorNum, anchors[idx]);
        if (shouldDetectLandmarks) {
            const auto landmarksRaw = infResult.outputsData[separateOutputsNames[OT_LANDMARK][idx]];
            filterLandmarks(landmarks, validIndices, landmarksRaw, anchorNum, anchors[idx], landmarkStd);
        }
        if (shouldDetectMasks) {
            const auto masksRaw = infResult.outputsData[separateOutputsNames[OT_MASKSCORES][idx]];
            filterMasksScores(masks, validIndices, masksRaw, anchorNum);
        }
    }
    // --------------------------- Apply Non-maximum Suppression ----------------------------------------------------------
    // !shouldDetectLandmarks determines nms behavior, if true - boundaries are included in areas calculation
    auto keep = nms(bboxes, scores, boxIOUThreshold, !shouldDetectLandmarks);

    // --------------------------- Create detection result objects --------------------------------------------------------
    RetinaFaceDetectionResult* result = new RetinaFaceDetectionResult(infResult.frameId, infResult.metaData);

    auto imgWidth = infResult.internalModelData->asRef<InternalImageModelData>().inputImgWidth;
    auto imgHeight = infResult.internalModelData->asRef<InternalImageModelData>().inputImgHeight;
    auto scaleX = static_cast<float>(netInputWidth) / imgWidth;
    auto scaleY = static_cast<float>(netInputHeight) / imgHeight;

    result->objects.reserve(keep.size());
    result->landmarks.reserve(keep.size() * ModelRetinaFace::LANDMARKS_NUM);
    for (auto i : keep) {
        DetectedObject desc;
        desc.confidence = scores[i];
        //--- Scaling coordinates
        bboxes[i].left /= scaleX;
        bboxes[i].top /= scaleY;
        bboxes[i].right /= scaleX;
        bboxes[i].bottom /= scaleY;

        desc.x = clamp(bboxes[i].left, 0.f, (float)imgWidth);
        desc.y = clamp(bboxes[i].top, 0.f, (float)imgHeight);
        desc.width = clamp(bboxes[i].getWidth(), 0.f, (float)imgWidth);
        desc.height = clamp(bboxes[i].getHeight(), 0.f, (float)imgHeight);
        //--- Default label 0 - Face. If detecting masks then labels would be 0 - No Mask, 1 - Mask
        desc.labelID = shouldDetectMasks ? (masks[i] > maskThreshold) : 0;
        desc.label = labels[desc.labelID];
        result->objects.push_back(desc);

        //--- Scaling landmarks coordinates
        for (size_t l = 0; l < ModelRetinaFace::LANDMARKS_NUM && shouldDetectLandmarks; ++l) {
            landmarks[i * ModelRetinaFace::LANDMARKS_NUM + l].x =
                clamp(landmarks[i * ModelRetinaFace::LANDMARKS_NUM + l].x / scaleX, 0.f, (float)imgWidth);
            landmarks[i * ModelRetinaFace::LANDMARKS_NUM + l].y =
                clamp(landmarks[i * ModelRetinaFace::LANDMARKS_NUM + l].y / scaleY, 0.f, (float)imgHeight);
            result->landmarks.push_back(landmarks[i * ModelRetinaFace::LANDMARKS_NUM + l]);
        }
    }

    return std::unique_ptr<ResultBase>(result);
}
