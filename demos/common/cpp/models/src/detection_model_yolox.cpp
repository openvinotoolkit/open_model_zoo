/*
// Copyright (C) 2022-2024 Intel Corporation
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

#include "models/detection_model_yolox.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>

#include <utils/common.hpp>
#include <utils/slog.hpp>

#include "models/input_data.h"
#include "models/internal_model_data.h"
#include "models/results.h"
#include "utils/image_utils.h"
#include "utils/nms.hpp"

ModelYoloX::ModelYoloX(const std::string& modelFileName,
                                 float confidenceThreshold,
                                 float boxIOUThreshold,
                                 const std::vector<std::string>& labels,
                                 const std::string& layout)
    : DetectionModel(modelFileName, confidenceThreshold, false, labels, layout),
      boxIOUThreshold(boxIOUThreshold) {
        resizeMode = RESIZE_KEEP_ASPECT;
}

void ModelYoloX::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input  ------------------------------------------------------
    const ov::OutputVector& inputs = model->inputs();
    if (inputs.size() != 1) {
        throw std::logic_error("YOLOX model wrapper accepts models that have only 1 input");
    }

    //--- Check image input
    const auto& input = model->input();
    const ov::Shape& inputShape = model->input().get_shape();
    ov::Layout inputLayout = getInputLayout(input);

    if (inputShape.size() != 4 && inputShape[ov::layout::channels_idx(inputLayout)] != 3) {
        throw std::logic_error("Expected 4D image input with 3 channels");
    }

    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout({"NHWC"});

    ppp.input().model().set_layout(inputLayout);

    //--- Reading image input parameters
    inputsNames.push_back(input.get_any_name());
    netInputWidth = inputShape[ov::layout::width_idx(inputLayout)];
    netInputHeight = inputShape[ov::layout::height_idx(inputLayout)];
    setStridesGrids();

    // --------------------------- Prepare output  -----------------------------------------------------
    if (model->outputs().size() != 1) {
        throw std::logic_error("YoloX model wrapper expects models that have only 1 output");
    }
    const auto& output = model->output();
    outputsNames.push_back(output.get_any_name());
    const ov::Shape& shape = output.get_shape();

    if (shape.size() != 3) {
        throw std::logic_error("YOLOX single output must have 3 dimensions, but had " + std::to_string(shape.size()));
    }
    ppp.output().tensor().set_element_type(ov::element::f32);

    model = ppp.build();
}

void ModelYoloX::setStridesGrids() {
    std::vector<size_t> strides = {8, 16, 32};
    std::vector<size_t> hsizes(3);
    std::vector<size_t> wsizes(3);

    for (size_t i = 0; i < strides.size(); ++i) {
        hsizes[i] = netInputHeight / strides[i];
        wsizes[i] = netInputWidth / strides[i];
    }

    for (size_t size_index = 0; size_index < hsizes.size(); ++size_index) {
        for (size_t h_index = 0; h_index < hsizes[size_index]; ++h_index) {
            for (size_t w_index = 0; w_index < wsizes[size_index]; ++w_index) {
                grids.emplace_back(w_index, h_index);
                expandedStrides.push_back(strides[size_index]);
            }
        }
    }
}

std::shared_ptr<InternalModelData> ModelYoloX::preprocess(const InputData& inputData,
                                                          ov::InferRequest& request) {
    const auto& origImg = inputData.asRef<ImageInputData>().inputImage;
    float scale = std::min(static_cast<float>(netInputWidth) / origImg.cols,
                           static_cast<float>(netInputHeight) / origImg.rows);

    cv::Mat resizedImage = resizeImageExt(origImg, netInputWidth, netInputHeight, resizeMode,
                                          interpolationMode, nullptr, cv::Scalar(114, 114, 114));

    request.set_input_tensor(wrapMat2Tensor(resizedImage));
    return std::make_shared<InternalScaleData>(origImg.cols, origImg.rows, scale, scale);
}

std::unique_ptr<ResultBase> ModelYoloX::postprocess(InferenceResult& infResult) {
    // Get metadata about input image shape and scale
    const auto& scale = infResult.internalModelData->asRef<InternalScaleData>();

    // Get output tensor
    const ov::Tensor& output = infResult.outputsData[outputsNames[0]];
    const auto& outputShape = output.get_shape();
    float* outputPtr = output.data<float>();

    // Generate detection results
    DetectionResult* result = new DetectionResult(infResult.frameId, infResult.metaData);

    // Update coordinates according to strides
    for (size_t box_index = 0; box_index < expandedStrides.size(); ++box_index) {
        size_t startPos = outputShape[2] * box_index;
        outputPtr[startPos] = (outputPtr[startPos] + grids[box_index].first) * expandedStrides[box_index];
        outputPtr[startPos + 1] = (outputPtr[startPos + 1] + grids[box_index].second) * expandedStrides[box_index];
        outputPtr[startPos + 2] = std::exp(outputPtr[startPos + 2]) * expandedStrides[box_index];
        outputPtr[startPos + 3] = std::exp(outputPtr[startPos + 3]) * expandedStrides[box_index];
    }

    // Filter predictions
    std::vector<Anchor> validBoxes;
    std::vector<float> scores;
    std::vector<size_t> classes;
    for (size_t box_index = 0; box_index < expandedStrides.size(); ++box_index) {
        size_t startPos = outputShape[2] * box_index;
        float score = outputPtr[startPos + 4];
        if (score < confidenceThreshold)
            continue;
        float maxClassScore = -1;
        size_t mainClass = 0;
        for (size_t class_index = 0; class_index < numberOfClasses; ++class_index) {
            if (outputPtr[startPos + 5 + class_index] > maxClassScore) {
                maxClassScore = outputPtr[startPos + 5 + class_index];
                mainClass = class_index;
            }
        }

        // Filter by score
        score *= maxClassScore;
        if (score < confidenceThreshold)
            continue;

        // Add successful boxes
        scores.push_back(score);
        classes.push_back(mainClass);
        Anchor trueBox = {outputPtr[startPos + 0] - outputPtr[startPos + 2] / 2, outputPtr[startPos + 1] - outputPtr[startPos + 3] / 2,
                          outputPtr[startPos + 0] + outputPtr[startPos + 2] / 2, outputPtr[startPos + 1] + outputPtr[startPos + 3] / 2};
        validBoxes.push_back(Anchor({trueBox.left / scale.scaleX, trueBox.top / scale.scaleY,
                                     trueBox.right / scale.scaleX, trueBox.bottom / scale.scaleY}));
    }

    // NMS for valid boxes
    std::vector<int> keep = nms(validBoxes, scores, boxIOUThreshold, true);
    for (auto& index: keep) {
        // Create new detected box
        DetectedObject obj;
        obj.x = clamp(validBoxes[index].left, 0.f, static_cast<float>(scale.inputImgWidth));
        obj.y = clamp(validBoxes[index].top, 0.f, static_cast<float>(scale.inputImgHeight));
        obj.height = clamp(validBoxes[index].bottom - validBoxes[index].top, 0.f, static_cast<float>(scale.inputImgHeight));
        obj.width = clamp(validBoxes[index].right - validBoxes[index].left, 0.f, static_cast<float>(scale.inputImgWidth));
        obj.confidence = scores[index];
        obj.labelID = classes[index];
        obj.label = getLabelName(classes[index]);
        result->objects.push_back(obj);
    }

    return std::unique_ptr<ResultBase>(result);
}
