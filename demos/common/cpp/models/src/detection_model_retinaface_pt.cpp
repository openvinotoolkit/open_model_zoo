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

#include <openvino/openvino.hpp>
#include <utils/common.hpp>
#include <utils/slog.hpp>

#include "models/detection_model_retinaface_pt.h"

ModelRetinaFacePT::ModelRetinaFacePT(const std::string& modelFileName, float confidenceThreshold, float boxIOUThreshold)
    : DetectionModel(modelFileName, confidenceThreshold, {"Face"}),  // Default label is "Face"
    landmarksNum(0), boxIOUThreshold(boxIOUThreshold) {
}

void ModelRetinaFacePT::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input ------------------------------------------------------
    if (model->inputs().size() != 1) {
        throw std::logic_error("RetinaFacePT model wrapper expects models that have only one input");
    }

    const ov::Shape& inputShape = model->input().get_shape();
    ov::Layout inputLayout = ov::layout::get_layout(model->input());
    if (inputLayout.empty()) {
        inputLayout = { "NCHW" };
    }

    if (inputShape[ov::layout::channels_idx(inputLayout)] != 3) {
        throw std::logic_error("Expected 3-channel input");
    }

    ov::preprocess::PrePostProcessor ppp(model);
    inputTransform.setPrecision(ppp, model->input().get_any_name());
    ppp.input().tensor().
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
    if (model->outputs().size() != 3) {
        throw std::logic_error("RetinaFace model wrapper expects models that have 3 outputs");
    }

    landmarksNum = 0;

    outputsNames.resize(2);
    std::vector<uint32_t> outputsSizes[OT_MAX];
    ov::Layout chw("CHW");
    ov::Layout nchw("NCHW");
    for (auto& output : model->outputs()) {
        auto outTensorName = output.get_any_name();
        outputsNames.push_back(outTensorName);
        ppp.output(outTensorName).tensor().
            set_element_type(ov::element::f32).
            set_layout(output.get_shape().size() == 4 ? nchw : chw);

        if (outTensorName.find("bbox") != std::string::npos) {
            outputsNames[OT_BBOX] = outTensorName;
        }
        else if (outTensorName.find("cls") != std::string::npos) {
            outputsNames[OT_SCORES] = outTensorName;
        }
        else if (outTensorName.find("landmark") != std::string::npos) {
            // Landmarks might be optional, if it is present, resize names array to fit landmarks output name to the last item of array
            // Considering that other outputs names are already filled in or will be filled later
            outputsNames.resize(std::max(outputsNames.size(), (size_t)OT_LANDMARK + 1));
            outputsNames[OT_LANDMARK] = outTensorName;
            landmarksNum = output.get_shape()[ov::layout::width_idx(chw)] / 2; // Each landmark consist of 2 variables (x and y)
        }
        else {
            continue;
        }
    }

    if (outputsNames[OT_BBOX] == "" || outputsNames[OT_SCORES] == "") {
        throw std::logic_error("Bbox or cls layers are not found");
    }

    model = ppp.build();
    priors = generatePriorData();
}


std::vector<size_t> ModelRetinaFacePT::filterByScore(const ov::Tensor& scoresTensor, const float confidenceThreshold) {
    std::vector<size_t> indicies;
    auto shape = scoresTensor.get_shape();
    const float* scoresPtr = scoresTensor.data<float>();

    for (size_t x = 0; x < shape[1]; ++x) {
        auto idx = (x * shape[2] + 1);
        auto score = scoresPtr[idx];
        if (score >= confidenceThreshold) {
            indicies.push_back(x);
        }
    }

    return indicies;
}

std::vector<float> ModelRetinaFacePT::getFilteredScores(const ov::Tensor& scoresTensor, const std::vector<size_t>& indicies) {
    auto shape = scoresTensor.get_shape();
    const float* scoresPtr = scoresTensor.data<float>();

    std::vector<float> scores;
    scores.reserve(indicies.size());

    for (auto i : indicies) {
        scores.push_back(scoresPtr[i*shape[2] + 1]);
    }
    return scores;
}

std::vector<cv::Point2f> ModelRetinaFacePT::getFilteredLandmarks(const ov::Tensor& landmarksTensor, const std::vector<size_t>& indicies, int imgWidth, int imgHeight) {
    auto shape = landmarksTensor.get_shape();
    const float* landmarksPtr = landmarksTensor.data<float>();

    std::vector<cv::Point2f> landmarks(landmarksNum*indicies.size());

    for (size_t i = 0; i < indicies.size(); i++) {
        size_t idx = indicies[i];
        auto& prior = priors[idx];
        for (size_t j = 0; j < landmarksNum; j++) {
            landmarks[i*landmarksNum + j].x = clamp(prior.cX + landmarksPtr[idx*shape[2] + j*2] * variance[0] * prior.width, 0.f, 1.f) * imgWidth;
            landmarks[i*landmarksNum + j].y = clamp(prior.cY + landmarksPtr[idx*shape[2] + j*2 + 1] * variance[0] * prior.height, 0.f, 1.f) * imgHeight;
        }
    }
    return landmarks;
}

std::vector<ModelRetinaFacePT::Box> ModelRetinaFacePT::generatePriorData() {
    float globalMinSizes[][2] = { {16, 32}, {64, 128}, {256, 512} };
    float steps[] = { 8., 16., 32. };
    std::vector<ModelRetinaFacePT::Box> anchors;
    for (size_t stepNum = 0; stepNum < arraySize(steps); stepNum++) {
        const int featureW = (int)std::round(netInputWidth / steps[stepNum]);
        const int featureH = (int)std::round(netInputHeight / steps[stepNum]);

        auto& minSizes = globalMinSizes[stepNum];
        for (int i = 0; i < featureH; i++) {
            for (int j = 0; j < featureW; j++) {
                for (auto minSize : minSizes) {
                    float sKX = minSize / netInputWidth;
                    float sKY = minSize / netInputHeight;
                    float denseCY = (i + 0.5f) * steps[stepNum] / netInputHeight;
                    float denseCX = (j + 0.5f) * steps[stepNum] / netInputWidth;
                    anchors.push_back(ModelRetinaFacePT::Box{denseCX, denseCY, sKX, sKY});
                }
            }
        }
    }
    return anchors;
}

std::vector<ModelRetinaFacePT::Rect> ModelRetinaFacePT::getFilteredProposals(const ov::Tensor& bboxesTensor, const std::vector<size_t>& indicies,int imgWidth, int imgHeight) {
    std::vector<ModelRetinaFacePT::Rect> rects;
    rects.reserve(indicies.size());

    auto shape = bboxesTensor.get_shape();
    const float* bboxesPtr = bboxesTensor.data<float>();


    if (shape[1] != priors.size()) {
        throw std::runtime_error("rawBoxes size is not equal to priors size");
    }

    for (auto i : indicies) {
        auto pRawBox = reinterpret_cast<const Box*>(bboxesPtr + i*shape[2]);
        auto& prior = priors[i];
        float cX = priors[i].cX + pRawBox->cX * variance[0] * prior.width;
        float cY = priors[i].cY + pRawBox->cY * variance[0] * prior.height;
        float width = prior.width * exp(pRawBox->width * variance[1]);
        float height = prior.height * exp(pRawBox->height * variance[1]);
        rects.push_back(Rect{
            clamp(cX - width / 2, 0.f, 1.f) * imgWidth,
            clamp(cY - height / 2, 0.f, 1.f) * imgHeight,
            clamp(cX + width / 2, 0.f, 1.f) * imgWidth,
            clamp(cY + height / 2, 0.f, 1.f) * imgHeight });
    }

    return rects;
}

std::unique_ptr<ResultBase> ModelRetinaFacePT::postprocess(InferenceResult& infResult) {
    //(raw_output, scale_x, scale_y, face_prob_threshold, image_size):
    const auto bboxesTensor = infResult.outputsData[outputsNames[OT_BBOX]];
    const auto scoresTensor = infResult.outputsData[outputsNames[OT_SCORES]];

    const auto& validIndicies = filterByScore(scoresTensor, confidenceThreshold);
    const auto& scores = getFilteredScores(scoresTensor, validIndicies);

    const auto& internalData = infResult.internalModelData->asRef<InternalImageModelData>();
    const auto& landmarks = landmarksNum ?
        getFilteredLandmarks(infResult.outputsData[outputsNames[OT_LANDMARK]], validIndicies, internalData.inputImgWidth, internalData.inputImgHeight) :
        std::vector<cv::Point2f>();

    const auto& proposals = getFilteredProposals(bboxesTensor, validIndicies, internalData.inputImgWidth, internalData.inputImgHeight);

    const auto& keptIndicies = nms(proposals, scores, boxIOUThreshold, !landmarksNum);

    // --------------------------- Create detection result objects --------------------------------------------------------
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
            result->landmarks.emplace_back(landmarks[i*landmarksNum +l].x,
                landmarks[i*landmarksNum + l].y
            );
        }
    }

    return std::unique_ptr<ResultBase>(result);
}
