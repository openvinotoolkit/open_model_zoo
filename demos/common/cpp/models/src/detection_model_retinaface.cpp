/*
// Copyright (C) 2020-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <ngraph/ngraph.hpp>
#include <utils/common.hpp>
#include <utils/slog.hpp>
#include "models/detection_model_retinaface.h"

ModelRetinaFace::ModelRetinaFace(const std::string& modelFileName, float confidenceThreshold, bool useAutoResize, float boxIOUThreshold)
    : DetectionModel(modelFileName, confidenceThreshold, useAutoResize, { "Face" }), // Default label is "Face"
    shouldDetectMasks(false), shouldDetectLandmarks(false), boxIOUThreshold(boxIOUThreshold), maskThreshold(0.8f), landmarkStd(1.0f),
    anchorCfg({ {32, { 32, 16 }, 16, { 1 }},
    { 16, { 8, 4 }, 16, { 1 }},
    { 8, { 2, 1 }, 16, { 1 }} }) {
    generateAnchorsFpn();
}

ModelBase::IOPattern ModelRetinaFace::getIOPattern() {
    ModelBase::BlobPattern inputPattern(
        "input",
        // Possible models' inputs
        // Describe number of inputs, precision, dimensions and layout.
        // If it doesn't matter what dimension's value is set 0.
        {
        { 1, { { "data", { InferenceEngine::Precision::U8, {1, 3, 0, 0}, useAutoResize ? InferenceEngine::Layout::NHWC : InferenceEngine::Layout::NCHW} } } }

        }
    );

    ModelBase::BlobPattern outputPattern(
        "output",
        // Possible models' outputs
        // Describe number of inputs, precision, dimensions and layout.
        // If it doesn't matter what dimension's value is - set 0.
        {
        { 6, { { "rpn_cls_prob_reshape_stride32", { InferenceEngine::Precision::FP32, {1, 4, 20, 20}, InferenceEngine::Layout::NCHW } },
        { "rpn_bbox_pred_stride32", { InferenceEngine::Precision::FP32, {1, 8, 20, 20}, InferenceEngine::Layout::NCHW } },
        { "rpn_cls_prob_reshape_stride16", { InferenceEngine::Precision::FP32, {1, 4, 40, 40}, InferenceEngine::Layout::NCHW } },
        { "rpn_bbox_pred_stride16", { InferenceEngine::Precision::FP32, {1, 8, 40, 40}, InferenceEngine::Layout::NCHW } },
        { "rpn_cls_prob_reshape_stride8", { InferenceEngine::Precision::FP32, {1, 4, 80, 80}, InferenceEngine::Layout::NCHW } },
        { "rpn_bbox_pred_stride8", { InferenceEngine::Precision::FP32, {1, 8, 80, 80}, InferenceEngine::Layout::NCHW } } } },

        { 9, { { "face_rpn_cls_prob_reshape_stride32", { InferenceEngine::Precision::FP32, {1, 4, 20, 20}, InferenceEngine::Layout::NCHW } },
        { "face_rpn_bbox_stride32", { InferenceEngine::Precision::FP32, {1, 8, 20, 20}, InferenceEngine::Layout::NCHW } },
        { "face_rpn_landmark_pred_stride32", { InferenceEngine::Precision::FP32, {1, 20, 20, 20}, InferenceEngine::Layout::NCHW } },
        { "face_rpn_cls_prob_reshape_stride16", { InferenceEngine::Precision::FP32, {1, 4, 40, 40}, InferenceEngine::Layout::NCHW } },
        { "face_rpn_bbox_pred_stride16", { InferenceEngine::Precision::FP32, {1, 8, 40, 40}, InferenceEngine::Layout::NCHW } },
        { "face_rpn_landmark_pred_stride16", { InferenceEngine::Precision::FP32, {1, 20, 40, 40}, InferenceEngine::Layout::NCHW } },
        { "face_rpn_cls_prob_reshape_stride8", { InferenceEngine::Precision::FP32, {1, 4, 80, 80}, InferenceEngine::Layout::NCHW } },
        { "face_rpn_bbox_pred_stride8", { InferenceEngine::Precision::FP32, {1, 8, 80, 80}, InferenceEngine::Layout::NCHW } },
        { "face_rpn_landmark_pred_stride8", { InferenceEngine::Precision::FP32, {1, 20, 80, 80}, InferenceEngine::Layout::NCHW } } } },

        { 12, { { "face_rpn_cls_prob_reshape_stride32", { InferenceEngine::Precision::FP32, {1, 4, 20, 20}, InferenceEngine::Layout::NCHW } },
        { "face_rpn_bbox_stride32", { InferenceEngine::Precision::FP32, {1, 8, 20, 20}, InferenceEngine::Layout::NCHW } },
        { "face_rpn_landmark_pred_stride32", { InferenceEngine::Precision::FP32, {1, 20, 20, 20}, InferenceEngine::Layout::NCHW } },
        { "face_rpn_type_prob_reshape_stride32", { InferenceEngine::Precision::FP32, {1, 6, 20, 20}, InferenceEngine::Layout::NCHW } },
        { "face_rpn_cls_prob_reshape_stride16", { InferenceEngine::Precision::FP32, {1, 4, 40, 40}, InferenceEngine::Layout::NCHW } },
        { "face_rpn_bbox_pred_stride16", { InferenceEngine::Precision::FP32, {1, 8, 40, 40}, InferenceEngine::Layout::NCHW } },
        { "face_rpn_landmark_pred_stride16", { InferenceEngine::Precision::FP32, {1, 20, 40, 40}, InferenceEngine::Layout::NCHW } },
        { "face_rpn_type_prob_reshape_stride16", { InferenceEngine::Precision::FP32, {1, 6, 40, 40}, InferenceEngine::Layout::NCHW } },
        { "face_rpn_cls_prob_reshape_stride8", { InferenceEngine::Precision::FP32, {1, 4, 80, 80}, InferenceEngine::Layout::NCHW } },
        { "face_rpn_bbox_pred_stride8", { InferenceEngine::Precision::FP32, {1, 8, 80, 80}, InferenceEngine::Layout::NCHW } },
        { "face_rpn_landmark_pred_stride8", { InferenceEngine::Precision::FP32, {1, 20, 80, 80}, InferenceEngine::Layout::NCHW } },
        { "face_rpn_type_prob_reshape_stride8", { InferenceEngine::Precision::FP32, {1, 6, 80, 80}, InferenceEngine::Layout::NCHW } } } },

        }
    );

    return { "RetinaFace", {inputPattern, outputPattern} };
}
template<class OutputsDataMap>
InferenceEngine::SizeVector ModelRetinaFace::getBoxexSizes(const OutputsDataMap& outputInfo) {
    std::vector<size_t> outputsSizes[OT_MAX];
    for (const auto& output : outputInfo) {

        outputsNames.push_back(output.first);

        EOutputType type = OT_MAX;
        if (output.first.find("bbox") != std::string::npos) {
            type = OT_BBOX;
        }
        else if (output.first.find("cls") != std::string::npos) {
            type = OT_SCORES;
        }
        else if (output.first.find("landmark") != std::string::npos) {
            type = OT_LANDMARK;
            shouldDetectLandmarks = true;
        }
        else if (output.first.find("type") != std::string::npos) {
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

        size_t num = output.second->getDims()[2];
        size_t i = 0;
        for (; i < outputsSizes[type].size(); ++i) {
            if (num < outputsSizes[type][i]) {
                break;
            }
        }
        separateOutputsNames[type].insert(separateOutputsNames[type].begin() + i, output.first);
        outputsSizes[type].insert(outputsSizes[type].begin() + i, num);
    }

    return outputsSizes[OT_BBOX];
}

void ModelRetinaFace::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output -------------------------------------------------
    ImageModel::prepareInputsOutputs(cnnNetwork);
    auto boxesSizes = getBoxexSizes(cnnNetwork.getOutputsInfo());
    calculatePriorBoxes(boxesSizes);
}

void ModelRetinaFace::checkCompiledNetworkInputsOutputs() {
    ImageModel::checkCompiledNetworkInputsOutputs();
    auto boxesSizes = getBoxexSizes(execNetwork.GetOutputsInfo());
    calculatePriorBoxes(boxesSizes);
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
        retVal.push_back({ static_cast<float>(xCtr - 0.5f * (ws - 1.0f)), static_cast<float>(yCtr - 0.5f * (hs - 1.0f)),
        static_cast<float>(xCtr + 0.5f * (ws - 1.0f)), static_cast<float>(yCtr + 0.5f * (hs - 1.0f)) });
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

void ModelRetinaFace::calculatePriorBoxes(const InferenceEngine::SizeVector& boxesSizes) {
    for (size_t idx = 0; idx < boxesSizes.size(); ++idx) {
        size_t width = boxesSizes[idx];
        size_t height = boxesSizes[idx];
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

std::vector<size_t> thresholding(const InferenceEngine::MemoryBlob::Ptr& rawData, const int anchorNum, const float confidenceThreshold) {
    std::vector<size_t> indices;
    indices.reserve(ModelRetinaFace::INIT_VECTOR_SIZE);
    auto desc = rawData->getTensorDesc();
    auto sz = desc.getDims();
    size_t restAnchors = sz[1] - anchorNum;
    InferenceEngine::LockedMemory<const void> outputMapped = rawData->rmap();
    const float* memPtr = outputMapped.as<float*>();

    for (size_t x = anchorNum; x < sz[1]; ++x) {
        for (size_t y = 0; y < sz[2]; ++y) {
            for (size_t z = 0; z < sz[3]; ++z) {
                auto idx = (x * sz[2] + y) * sz[3] + z;
                auto score = memPtr[idx];
                if (score >= confidenceThreshold) {
                    indices.push_back((y * sz[3] + z) * restAnchors + (x - anchorNum));
                }
            }
        }
    }

    return indices;
}

void filterScores(std::vector<float>& scores, const std::vector<size_t>& indices, const InferenceEngine::MemoryBlob::Ptr& rawData, const int anchorNum) {
    InferenceEngine::LockedMemory<const void> outputMapped = rawData->rmap();
    const float* memPtr = outputMapped.as<float*>();
    auto desc = rawData->getTensorDesc();
    auto sz = desc.getDims();
    auto start = sz[2] * sz[3] * anchorNum;

    for (auto i : indices) {
        auto offset = (i % anchorNum) * sz[2] * sz[3] + i / anchorNum;;
        scores.push_back(memPtr[start + offset]);
    }
}

void filterBBoxes(std::vector<ModelRetinaFace::Anchor>& bboxes, const std::vector<size_t>& indices, const InferenceEngine::MemoryBlob::Ptr& rawData,
    int anchorNum, const std::vector<ModelRetinaFace::Anchor>& anchors) {
    auto desc = rawData->getTensorDesc();
    auto sz = desc.getDims();
    auto bboxPredLen = sz[1] / anchorNum;
    auto blockWidth = sz[2] * sz[3];
    InferenceEngine::LockedMemory<const void> outputMapped = rawData->rmap();
    const float* memPtr = outputMapped.as<float*>();

    for (auto i : indices) {
        auto offset = blockWidth * bboxPredLen * (i % anchorNum) + (i / anchorNum);

        auto dx = memPtr[offset];
        auto dy = memPtr[offset + blockWidth];
        auto dw = memPtr[offset + blockWidth * 2];
        auto dh = memPtr[offset + blockWidth * 3];

        auto predCtrX = dx * anchors[i].getWidth() + anchors[i].getXCenter();
        auto predCtrY = dy * anchors[i].getHeight() + anchors[i].getYCenter();
        auto predW = exp(dw) * anchors[i].getWidth();
        auto predH = exp(dh) * anchors[i].getHeight();

        bboxes.push_back({ static_cast<float>(predCtrX - 0.5f * (predW - 1.0f)), static_cast<float>(predCtrY - 0.5f * (predH - 1.0f)),
        static_cast<float>(predCtrX + 0.5f * (predW - 1.0f)), static_cast<float>(predCtrY + 0.5f * (predH - 1.0f)) });
    }
}


void filterLandmarks(std::vector<cv::Point2f>& landmarks, const std::vector<size_t>& indices, const InferenceEngine::MemoryBlob::Ptr& rawData,
    int anchorNum, const std::vector<ModelRetinaFace::Anchor>& anchors, const float landmarkStd) {
    auto desc = rawData->getTensorDesc();
    auto sz = desc.getDims();
    auto landmarkPredLen = sz[1] / anchorNum;
    auto blockWidth = sz[2] * sz[3];
    InferenceEngine::LockedMemory<const void> outputMapped = rawData->rmap();
    const float* memPtr = outputMapped.as<float*>();

    for (auto i : indices) {
        for (int j = 0; j < ModelRetinaFace::LANDMARKS_NUM; ++j) {
            auto offset = (i % anchorNum) * landmarkPredLen * sz[2] * sz[3] + i / anchorNum;
            auto deltaX = memPtr[offset + j * 2 * blockWidth] * landmarkStd;
            auto deltaY = memPtr[offset + (j * 2 + 1) * blockWidth] * landmarkStd;
            landmarks.push_back({ deltaX * anchors[i].getWidth() + anchors[i].getXCenter(),
            deltaY * anchors[i].getHeight() + anchors[i].getYCenter() });
        }
    }
}

void filterMasksScores(std::vector<float>& masks, const std::vector<size_t>& indices, const InferenceEngine::MemoryBlob::Ptr& rawData, const int anchorNum) {
    auto desc = rawData->getTensorDesc();
    auto sz = desc.getDims();
    auto start = sz[2] * sz[3] * anchorNum * 2;
    InferenceEngine::LockedMemory<const void> outputMapped = rawData->rmap();
    const float* memPtr = outputMapped.as<float*>();

    for (auto i : indices) {
        auto offset = (i % anchorNum) * sz[2] * sz[3] + i / anchorNum;
        masks.push_back(memPtr[start + offset]);
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
    // !shouldDetectLandmarks determines nms behavior, if true - boundaries are included in areas calculation for ssh-mxnet model
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

        desc.x = bboxes[i].left;
        desc.y = bboxes[i].top;
        desc.width = bboxes[i].getWidth();
        desc.height = bboxes[i].getHeight();
        //--- Default label 0 - Face. If detecting masks then labels would be 0 - No Mask, 1 - Mask
        desc.labelID = shouldDetectMasks ? (masks[i] > maskThreshold) : 0;
        desc.label = labels[desc.labelID];
        result->objects.push_back(desc);

        //--- Scaling landmarks coordinates
        for (size_t l = 0; l < ModelRetinaFace::LANDMARKS_NUM && shouldDetectLandmarks; ++l) {
            landmarks[i * ModelRetinaFace::LANDMARKS_NUM + l].x /= scaleX;
            landmarks[i * ModelRetinaFace::LANDMARKS_NUM + l].y /= scaleY;
            result->landmarks.push_back(landmarks[i * ModelRetinaFace::LANDMARKS_NUM + l]);
        }
    }

    return std::unique_ptr<ResultBase>(result);;
}
