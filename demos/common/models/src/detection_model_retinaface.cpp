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

#include "models/detection_model_retinaface.h"
#include <ngraph/ngraph.hpp>
#include <samples/slog.hpp>
#include <samples/common.hpp>

using namespace InferenceEngine;

ModelRetinaFace::ModelRetinaFace(const std::string& modelFileName, float confidenceThreshold, bool useAutoResize)
    : DetectionModel(modelFileName, confidenceThreshold, useAutoResize, { "Face" }),
    anchorCfg({ {32, { 32, 16 }, 16, { 1.0 }}, { 16, { 8, 4 }, 16, { 1.0 }}, { 8, { 2, 1 }, 16, { 1.0 }} }) {
    generateAnchorsFpn();
    iouThreshold = 0.5;
    maskThreshold = 0.7;
    shouldDetectMasks = false;
    landmarkStd = 1.0;
}

void ModelRetinaFace::prepareInputsOutputs(InferenceEngine::CNNNetwork & cnnNetwork) {
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

    std::vector<int> outputsSizes[OT_MAX];
    for (auto& output : outputInfo) {
        output.second->setPrecision(InferenceEngine::Precision::FP32);
        output.second->setLayout(InferenceEngine::Layout::NCHW);
        outputsNames.push_back(output.first);

        EOutputType type= OT_MAX;
        if (output.first.find("bbox") != -1) {
            type = OT_BBOX;
        }
        else if (output.first.find("cls") != -1) {
            type = OT_SCORES;
        }
        else if(output.first.find("landmark") != -1) {
            type = OT_LANDMARK;
        }
        else if(output.first.find("type") != -1) {
            type = OT_MASKSCORES;
            labels.clear();
            labels.push_back("Mask");
            labels.push_back("No Mask");
            shouldDetectMasks = true;
            landmarkStd = 0.2;
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
        separateOutputsNames[type].insert(separateOutputsNames[type].begin()+i,output.first);
        outputsSizes[type].insert(outputsSizes[type].begin() + i, num);
    }

    if (outputsNames.size()!=9 && outputsNames.size() != 12)
        throw std::logic_error("Expected 12 or 9 output blobs");
}

std::vector<ModelRetinaFace::Anchor> _ratio_enum(const ModelRetinaFace::Anchor& anchor, std::vector<double> ratios) {
    std::vector<ModelRetinaFace::Anchor> retVal;
    auto w = anchor.getWidth();
    auto h = anchor.getHeight();
    auto xCtr = anchor.getXCenter();
    auto yCtr = anchor.getYCenter();
    for (auto ratio : ratios) {
        auto size = w * h;
        auto size_ratio = size / ratio;
        auto ws = std::round(sqrt(size_ratio));
        auto hs = std::round(ws * ratio);
        retVal.push_back({ xCtr - 0.5 * (ws - 1), yCtr - 0.5 * (hs - 1), xCtr + 0.5 * (ws - 1), yCtr + 0.5 * (hs - 1) });
    }
    return retVal;
}

std::vector<ModelRetinaFace::Anchor> _scale_enum(const ModelRetinaFace::Anchor& anchor, std::vector<double> scales) {
    std::vector<ModelRetinaFace::Anchor> retVal;
    auto w = anchor.getWidth();
    auto h = anchor.getHeight();
    auto xCtr = anchor.getXCenter();
    auto yCtr = anchor.getYCenter();
    for (auto scale : scales) {
        auto ws = w * scale;
        auto hs = h * scale;
        retVal.push_back({ xCtr - 0.5 * (ws - 1), yCtr - 0.5 * (hs - 1), xCtr + 0.5 * (ws - 1), yCtr + 0.5 * (hs - 1) });
    }
    return retVal;
}

std::vector<ModelRetinaFace::Anchor> generate_anchors(int base_size, const std::vector<double>& ratios, const std::vector<double>& scales) {
    ModelRetinaFace::Anchor base_anchor{ 0, 0, static_cast<double>(base_size) - 1, static_cast<double>(base_size) - 1 };
    auto ratio_anchors = _ratio_enum(base_anchor, ratios);
    std::vector<ModelRetinaFace::Anchor> retVal;

    for (auto ra : ratio_anchors) {
        auto addon = _scale_enum(ra, scales);
        retVal.insert(retVal.end(), addon.begin(), addon.end());
    }
    return retVal;
}

void ModelRetinaFace::generateAnchorsFpn() {
    auto cfg = anchorCfg;
    std::sort(cfg.begin(), cfg.end(), [](AnchorCfgLine& x, AnchorCfgLine& y) {return x.stride > y.stride; });

    std::vector<ModelRetinaFace::Anchor> anchors;
    for (auto cfgLine : cfg) {
        auto anchors = generate_anchors(cfgLine.baseSize, cfgLine.ratios, cfgLine.scales);
        _anchors_fpn.emplace(cfgLine.stride,anchors);
    }
}


std::vector<size_t> _get_indices(InferenceEngine::MemoryBlob::Ptr rawData, int anchor_num, double confidenceThreshold) {
    std::vector<size_t> indices;
    auto desc = rawData->getTensorDesc();
    auto sz = desc.getDims();

    size_t restAnchors = sz[1] - anchor_num;
    LockedMemory<const void> outputMapped = rawData->rmap();
    const float *memPtr = outputMapped.as<float*>();

    for (size_t x = anchor_num; x < sz[1]; ++x) {
        for (size_t y = 0; y < sz[2]; ++y) {
            for (size_t z = 0; z < sz[3]; ++z) {
                auto idx = (x*sz[2] + y)*sz[3] + z;
                auto score = memPtr[idx];
                if (score >= confidenceThreshold) {
                    indices.push_back((y*sz[3] + z)*restAnchors + (x - anchor_num));
                }
            }
        }
    }

    return indices;
}

void _filter_scores(std::vector<double>* scores, const std::vector<size_t>& indices, InferenceEngine::MemoryBlob::Ptr rawData, int anchor_num) {
    auto desc = rawData->getTensorDesc();
    auto sz = desc.getDims();

    size_t restAnchors = sz[1] - anchor_num;
    LockedMemory<const void> outputMapped = rawData->rmap();
    const float *memPtr = outputMapped.as<float*>();

    for (auto i : indices) {
        scores->push_back(memPtr[i]);
    }
}

void _filter_bboxes(std::vector<ModelRetinaFace::Anchor>* bboxes, const std::vector<size_t>& indices, InferenceEngine::MemoryBlob::Ptr rawData,
    int anchor_num, std::vector<ModelRetinaFace::Anchor>& anchors) {
    auto desc = rawData->getTensorDesc();
    auto sz = desc.getDims();
    LockedMemory<const void> outputMapped = rawData->rmap();
    const float *memPtr = outputMapped.as<float*>();
    auto bbox_pred_len = sz[1] / anchor_num;
    auto blockWidth = sz[2] * sz[3];

    for (auto i : indices) {
        auto offset = blockWidth * bbox_pred_len * (i % anchor_num) + (i / anchor_num);

        auto dx = memPtr[offset];
        auto dy = memPtr[offset + blockWidth];
        auto dw = memPtr[offset + blockWidth * 2];
        auto dh = memPtr[offset + blockWidth * 3];

        auto pred_ctr_x = dx * anchors[i].getWidth() + anchors[i].getXCenter();
        auto pred_ctr_y = dy * anchors[i].getHeight() + anchors[i].getYCenter();
        auto pred_w = exp(dw) * anchors[i].getWidth();
        auto pred_h = exp(dh) * anchors[i].getHeight();

        bboxes->push_back({ pred_ctr_x - 0.5 * (pred_w - 1.0), pred_ctr_y - 0.5 * (pred_h - 1.0),
            pred_ctr_x + 0.5 * (pred_w - 1.0), pred_ctr_y + 0.5 * (pred_h - 1.0) });
    }
}


void _filter_landmarks(std::vector<cv::Point2f>* landmarks, const std::vector<size_t>& indices, InferenceEngine::MemoryBlob::Ptr rawData,
    int anchor_num, const std::vector<ModelRetinaFace::Anchor>& anchors, double landmarkStd) {
    auto desc = rawData->getTensorDesc();
    auto sz = desc.getDims();
    LockedMemory<const void> outputMapped = rawData->rmap();
    const float *memPtr = outputMapped.as<float*>();
    auto landmark_pred_len = sz[1] / anchor_num;
    auto stride = landmark_pred_len * sz[2] * sz[3];

    for (auto i : indices) {
        auto ctrX = anchors[i].getXCenter();
        auto ctrY = anchors[i].getYCenter();
        auto blockWidth = sz[2] * sz[3];
        for (int j = 0; j < ModelRetinaFace::LANDMARKS_NUM; ++j) {
            auto deltaX = (i % 2 ? memPtr[stride + i / 2 + j * 2 * blockWidth] : memPtr[i / 2 + j * 2 * blockWidth]) * landmarkStd;
            auto deltaY = (i % 2 ? memPtr[stride + i / 2 + (j * 2 + 1)*blockWidth] : memPtr[i / 2 + (j * 2 + 1)*blockWidth]) *  landmarkStd;
            landmarks->push_back({ static_cast<float>(deltaX * anchors[i].getWidth() + anchors[i].getXCenter()),
            static_cast<float>(deltaY * anchors[i].getHeight() + anchors[i].getYCenter()) });
        }
    }
}

void _filter_masks_scores(std::vector<double>* masks, const std::vector<size_t>& indices, InferenceEngine::MemoryBlob::Ptr rawData, int anchor_num) {
    auto desc = rawData->getTensorDesc();
    auto sz = desc.getDims();

    size_t restAnchors = sz[1] - anchor_num * 2;
    std::vector<double> retVal(restAnchors*sz[2] * sz[3]);

    LockedMemory<const void> outputMapped = rawData->rmap();
    const float *memPtr = outputMapped.as<float*>();

    for (auto i : indices) {
        masks->push_back(memPtr[i]);
    }
}

std::vector<int> nms(const std::vector<ModelRetinaFace::Anchor>& boxes, const std::vector<double>& scores, double thresh) {
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

std::unique_ptr<ResultBase>  ModelRetinaFace::postprocess(InferenceResult& infResult) {
    std::vector<double> scores;
    scores.reserve(INIT_VECTOR_SIZE);
    std::vector<Anchor> bboxes;
    bboxes.reserve(INIT_VECTOR_SIZE);
    std::vector<cv::Point2f> landmarks;
    landmarks.reserve(INIT_VECTOR_SIZE);
    std::vector<size_t> valid_indices;
    valid_indices.reserve(INIT_VECTOR_SIZE);
    std::vector<double> masks;
    if (shouldDetectMasks) {
        masks.reserve(INIT_VECTOR_SIZE);
    }
    for (int idx = 0; idx < anchorCfg.size(); ++idx) {
        auto s = anchorCfg[idx].stride;
        auto anchors_fpn = _anchors_fpn[s];
        auto anchor_num = anchors_fpn.size();
        const auto bbox_deltas = infResult.outputsData[separateOutputsNames[OT_BBOX][idx]];
        const auto scores_raw = infResult.outputsData[separateOutputsNames[OT_SCORES][idx]];
        const auto landmarks_raw = infResult.outputsData[separateOutputsNames[OT_LANDMARK][idx]];
        auto sz = bbox_deltas->getTensorDesc().getDims();
        auto height = sz[2];
        auto width = sz[3];

        //--- Creating strided anchors plane
        std::vector<Anchor> anchors(height * width * anchor_num);

        for (int iw = 0; iw < width; ++iw) {
            auto sw = iw * s;
            for (int ih = 0; ih < height; ++ih) {
                auto sh = ih * s;
                for (int k = 0; k < anchor_num; ++k) {
                    Anchor& anc = anchors[(ih * width + iw) * anchor_num + k];
                    anc.left = anchors_fpn[k].left + sw;
                    anc.top = anchors_fpn[k].top + sh;
                    anc.right = anchors_fpn[k].right + sw;
                    anc.bottom = anchors_fpn[k].bottom + sh;
                }
            }
        }
        valid_indices.clear();
        valid_indices = _get_indices(scores_raw, anchor_num, confidenceThreshold);
        _filter_scores(&scores, valid_indices, scores_raw, anchor_num);
        _filter_bboxes(&bboxes, valid_indices, bbox_deltas, anchor_num, anchors);
        _filter_landmarks(&landmarks, valid_indices, landmarks_raw, anchor_num, anchors, landmarkStd);
        if (shouldDetectMasks) {
            const auto masks_raw = infResult.outputsData[separateOutputsNames[OT_MASKSCORES][idx]];
            _filter_masks_scores(&masks, valid_indices, masks_raw, anchor_num);
        }
    }

    auto keep = scores.size() ? nms(bboxes, scores, 0.5) : std::vector<int>();
    RetinaFaceDetectionResult* result = new RetinaFaceDetectionResult;
    *static_cast<ResultBase*>(result) = static_cast<ResultBase&>(infResult);

    auto img_width = infResult.internalModelData->asRef<InternalImageModelData>().inputImgWidth;
    auto img_height = infResult.internalModelData->asRef<InternalImageModelData>().inputImgHeight;
    double scale_x = ((double)netInputWidth) / img_width;
    double scale_y = ((double)netInputHeight) / img_height;

    result->objects.reserve(keep.size());
    result->landmarks.reserve(keep.size());
    if (shouldDetectMasks) {
        result->masks.reserve(keep.size());
    }
    for (auto i : keep) {
        DetectedObject desc;
        desc.confidence = static_cast<float>(scores[i]);
        desc.x = static_cast<float>(bboxes[i].left / scale_x);
        desc.y = static_cast<float>(bboxes[i].top / scale_y);
        desc.width = static_cast<float>(bboxes[i].getWidth() / scale_x);
        desc.height = static_cast<float>(bboxes[i].getHeight() / scale_y);
        desc.labelID = shouldDetectMasks ? (masks[i] > maskThreshold) : 0;
        desc.label = labels[desc.labelID];
        result->objects.push_back(desc);

        /** scaling landmarks coordinates **/
        for (size_t l = 0; l < ModelRetinaFace::LANDMARKS_NUM; ++l) {
            landmarks[i * ModelRetinaFace::LANDMARKS_NUM + l].x /= scale_x;
            landmarks[i * ModelRetinaFace::LANDMARKS_NUM + l].y /= scale_y;
            result->landmarks.push_back(landmarks[i * ModelRetinaFace::LANDMARKS_NUM + l]);
        }
    }

    return std::unique_ptr<ResultBase>(result);;
}
