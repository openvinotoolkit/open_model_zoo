/*
// Copyright (C) 2021-2022 Intel Corporation
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

#include "models/detection_model_retinaface_pt.h"

#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

#include <utils/common.hpp>
#include <utils/nms.hpp>
#include <utils/ocv_common.hpp>

#include "models/internal_model_data.h"
#include "models/results.h"

ModelRetinaFacePT::ModelRetinaFacePT(const std::string& modelFileName,
                                     float confidenceThreshold,
                                     bool useAutoResize,
                                     float boxIOUThreshold,
                                     const std::string& layout)
    : DetectionModel(modelFileName, confidenceThreshold, useAutoResize, {"Face"}, layout),  // Default label is "Face"
      landmarksNum(0),
      boxIOUThreshold(boxIOUThreshold) {}

void ModelRetinaFacePT::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input ------------------------------------------------------
    if (model->inputs().size() != 1) {
        throw std::logic_error("RetinaFacePT model wrapper expects models that have only 1 input");
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
    if (model->outputs().size() != 3) {
        throw std::logic_error("RetinaFace model wrapper expects models that have 3 outputs");
    }

    landmarksNum = 0;

    outputsNames.resize(2);
    std::vector<uint32_t> outputsSizes[OUT_MAX];
    const ov::Layout chw("CHW");
    const ov::Layout nchw("NCHW");
    for (auto& output : model->outputs()) {
        auto outTensorName = output.get_any_name();
        outputsNames.push_back(outTensorName);
        ppp.output(outTensorName)
            .tensor()
            .set_element_type(ov::element::f32)
            .set_layout(output.get_shape().size() == 4 ? nchw : chw);

        if (outTensorName.find("bbox") != std::string::npos) {
            outputsNames[OUT_BOXES] = outTensorName;
        } else if (outTensorName.find("cls") != std::string::npos) {
            outputsNames[OUT_SCORES] = outTensorName;
        } else if (outTensorName.find("landmark") != std::string::npos) {
            // Landmarks might be optional, if it is present, resize names array to fit landmarks output name to the
            // last item of array Considering that other outputs names are already filled in or will be filled later
            outputsNames.resize(std::max(outputsNames.size(), (size_t)OUT_LANDMARKS + 1));
            outputsNames[OUT_LANDMARKS] = outTensorName;
            landmarksNum =
                output.get_shape()[ov::layout::width_idx(chw)] / 2;  // Each landmark consist of 2 variables (x and y)
        } else {
            continue;
        }
    }

    if (outputsNames[OUT_BOXES] == "" || outputsNames[OUT_SCORES] == "") {
        throw std::logic_error("Bbox or cls layers are not found");
    }

    model = ppp.build();
    priors = generatePriorData();
}

std::vector<size_t> ModelRetinaFacePT::filterByScore(const ov::Tensor& scoresTensor, const float confidenceThreshold) {
    std::vector<size_t> indicies;
    const auto& shape = scoresTensor.get_shape();
    const float* scoresPtr = scoresTensor.data<float>();

    for (size_t x = 0; x < shape[1]; ++x) {
        const auto idx = (x * shape[2] + 1);
        const auto score = scoresPtr[idx];
        if (score >= confidenceThreshold) {
            indicies.push_back(x);
        }
    }

    return indicies;
}

std::vector<float> ModelRetinaFacePT::getFilteredScores(const ov::Tensor& scoresTensor,
                                                        const std::vector<size_t>& indicies) {
    const auto& shape = scoresTensor.get_shape();
    const float* scoresPtr = scoresTensor.data<float>();

    std::vector<float> scores;
    scores.reserve(indicies.size());

    for (auto i : indicies) {
        scores.push_back(scoresPtr[i * shape[2] + 1]);
    }
    return scores;
}

std::vector<cv::Point2f> ModelRetinaFacePT::getFilteredLandmarks(const ov::Tensor& landmarksTensor,
                                                                 const std::vector<size_t>& indicies,
                                                                 int imgWidth,
                                                                 int imgHeight) {
    const auto& shape = landmarksTensor.get_shape();
    const float* landmarksPtr = landmarksTensor.data<float>();

    std::vector<cv::Point2f> landmarks(landmarksNum * indicies.size());

    for (size_t i = 0; i < indicies.size(); i++) {
        const size_t idx = indicies[i];
        const auto& prior = priors[idx];
        for (size_t j = 0; j < landmarksNum; j++) {
            landmarks[i * landmarksNum + j].x =
                clamp(prior.cX + landmarksPtr[idx * shape[2] + j * 2] * variance[0] * prior.width, 0.f, 1.f) * imgWidth;
            landmarks[i * landmarksNum + j].y =
                clamp(prior.cY + landmarksPtr[idx * shape[2] + j * 2 + 1] * variance[0] * prior.height, 0.f, 1.f) *
                imgHeight;
        }
    }
    return landmarks;
}

std::vector<ModelRetinaFacePT::Box> ModelRetinaFacePT::generatePriorData() {
    const float globalMinSizes[][2] = {{16, 32}, {64, 128}, {256, 512}};
    const float steps[] = {8., 16., 32.};
    std::vector<ModelRetinaFacePT::Box> anchors;
    for (size_t stepNum = 0; stepNum < arraySize(steps); stepNum++) {
        const int featureW = static_cast<int>(std::round(netInputWidth / steps[stepNum]));
        const int featureH = static_cast<int>(std::round(netInputHeight / steps[stepNum]));

        const auto& minSizes = globalMinSizes[stepNum];
        for (int i = 0; i < featureH; i++) {
            for (int j = 0; j < featureW; j++) {
                for (auto minSize : minSizes) {
                    const float sKX = minSize / netInputWidth;
                    const float sKY = minSize / netInputHeight;
                    const float denseCY = (i + 0.5f) * steps[stepNum] / netInputHeight;
                    const float denseCX = (j + 0.5f) * steps[stepNum] / netInputWidth;
                    anchors.push_back(ModelRetinaFacePT::Box{denseCX, denseCY, sKX, sKY});
                }
            }
        }
    }
    return anchors;
}

std::vector<ModelRetinaFacePT::Rect> ModelRetinaFacePT::getFilteredProposals(const ov::Tensor& boxesTensor,
                                                                             const std::vector<size_t>& indicies,
                                                                             int imgWidth,
                                                                             int imgHeight) {
    std::vector<ModelRetinaFacePT::Rect> rects;
    rects.reserve(indicies.size());

    const auto& shape = boxesTensor.get_shape();
    const float* boxesPtr = boxesTensor.data<float>();

    if (shape[1] != priors.size()) {
        throw std::logic_error("rawBoxes size is not equal to priors size");
    }

    for (auto i : indicies) {
        const auto pRawBox = reinterpret_cast<const Box*>(boxesPtr + i * shape[2]);
        const auto& prior = priors[i];
        const float cX = priors[i].cX + pRawBox->cX * variance[0] * prior.width;
        const float cY = priors[i].cY + pRawBox->cY * variance[0] * prior.height;
        const float width = prior.width * exp(pRawBox->width * variance[1]);
        const float height = prior.height * exp(pRawBox->height * variance[1]);
        rects.push_back(Rect{clamp(cX - width / 2, 0.f, 1.f) * imgWidth,
                             clamp(cY - height / 2, 0.f, 1.f) * imgHeight,
                             clamp(cX + width / 2, 0.f, 1.f) * imgWidth,
                             clamp(cY + height / 2, 0.f, 1.f) * imgHeight});
    }

    return rects;
}

std::unique_ptr<ResultBase> ModelRetinaFacePT::postprocess(InferenceResult& infResult) {
    // (raw_output, scale_x, scale_y, face_prob_threshold, image_size):
    const auto boxesTensor = infResult.outputsData[outputsNames[OUT_BOXES]];
    const auto scoresTensor = infResult.outputsData[outputsNames[OUT_SCORES]];

    const auto& validIndicies = filterByScore(scoresTensor, confidenceThreshold);
    const auto& scores = getFilteredScores(scoresTensor, validIndicies);

    const auto& internalData = infResult.internalModelData->asRef<InternalImageModelData>();
    const auto& landmarks = landmarksNum ? getFilteredLandmarks(infResult.outputsData[outputsNames[OUT_LANDMARKS]],
                                                                validIndicies,
                                                                internalData.inputImgWidth,
                                                                internalData.inputImgHeight)
                                         : std::vector<cv::Point2f>();

    const auto& proposals =
        getFilteredProposals(boxesTensor, validIndicies, internalData.inputImgWidth, internalData.inputImgHeight);

    const auto& keptIndicies = nms(proposals, scores, boxIOUThreshold, !landmarksNum);

    // --------------------------- Create detection result objects
    // --------------------------------------------------------
    RetinaFaceDetectionResult* result = new RetinaFaceDetectionResult(infResult.frameId, infResult.metaData);

    result->objects.reserve(keptIndicies.size());
    result->landmarks.reserve(keptIndicies.size() * landmarksNum);
    for (auto i : keptIndicies) {
        DetectedObject desc;
        desc.confidence = scores[i];

        //--- Scaling coordinates
        desc.x = proposals[i].left;
        desc.y = proposals[i].top;
        desc.width = proposals[i].getWidth();
        desc.height = proposals[i].getHeight();

        desc.labelID = 0;
        desc.label = labels[desc.labelID];
        result->objects.push_back(desc);

        //--- Filtering landmarks coordinates
        for (uint32_t l = 0; l < landmarksNum; ++l) {
            result->landmarks.emplace_back(landmarks[i * landmarksNum + l].x, landmarks[i * landmarksNum + l].y);
        }
    }

    return std::unique_ptr<ResultBase>(result);
}
