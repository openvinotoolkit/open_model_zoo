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

#include "models/detection_model_yolov3_onnx.h"

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

ModelYoloV3ONNX::ModelYoloV3ONNX(const std::string& modelFileName,
                                 float confidenceThreshold,
                                 const std::vector<std::string>& labels,
                                 const std::string& layout)
    : DetectionModel(modelFileName, confidenceThreshold, false, labels, layout) {
        interpolationMode = cv::INTER_CUBIC;
        resizeMode = RESIZE_KEEP_ASPECT_LETTERBOX;
    }


void ModelYoloV3ONNX::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare inputs ------------------------------------------------------
    const ov::OutputVector& inputs = model->inputs();
    if (inputs.size() != 2) {
        throw std::logic_error("YoloV3ONNX model wrapper expects models that have 2 inputs");
    }

    ov::preprocess::PrePostProcessor ppp(model);
    inputsNames.reserve(inputs.size());
    for (auto& input : inputs) {
        const ov::Shape& currentShape = input.get_shape();
        std::string currentName = input.get_any_name();
        const ov::Layout& currentLayout = getInputLayout(input);

        if (currentShape.size() == 4) {
            if (currentShape[ov::layout::channels_idx(currentLayout)] != 3) {
                throw std::logic_error("Expected 4D image input with 3 channels");
            }
            inputsNames[0] = currentName;
            netInputWidth = currentShape[ov::layout::width_idx(currentLayout)];
            netInputHeight = currentShape[ov::layout::height_idx(currentLayout)];
            ppp.input(currentName).tensor().set_element_type(ov::element::u8).set_layout({"NHWC"});
        } else if (currentShape.size() == 2) {
            if (currentShape[ov::layout::channels_idx(currentLayout)] != 2) {
                throw std::logic_error("Expected 2D image info input with 2 channels");
            }
            inputsNames[1] = currentName;
            ppp.input(currentName).tensor().set_element_type(ov::element::i32);
        }
        ppp.input(currentName).model().set_layout(currentLayout);
    }

    // --------------------------- Prepare outputs -----------------------------------------------------
    const ov::OutputVector& outputs = model->outputs();
    if (outputs.size() != 3) {
        throw std::logic_error("YoloV3ONNX model wrapper expects models that have 3 outputs");
    }

    for (auto& output : outputs) {
        const ov::Shape& currentShape = output.get_partial_shape().get_max_shape();
        std::string currentName = output.get_any_name();
        if (currentShape.back() == 3) {
            indicesOutputName = currentName;
            ppp.output(currentName).tensor().set_element_type(ov::element::i32);
        } else if (currentShape[2] == 4) {
            boxesOutputName = currentName;
            ppp.output(currentName).tensor().set_element_type(ov::element::f32);
        } else if (currentShape[1] == numberOfClasses) {
            scoresOutputName = currentName;
            ppp.output(currentName).tensor().set_element_type(ov::element::f32);
        } else {
            throw std::logic_error("Expected shapes [:,:,4], [:,"
                + std::to_string(numberOfClasses) + ",:] and [:,3] for outputs");
        }
        outputsNames.push_back(currentName);
    }
    model = ppp.build();
}

std::shared_ptr<InternalModelData> ModelYoloV3ONNX::preprocess(const InputData& inputData,
                                                               ov::InferRequest& request) {
    const auto& origImg = inputData.asRef<ImageInputData>().inputImage;
    ov::Tensor info{ov::element::i32, ov::Shape({1, 2})};
    int32_t* data = info.data<int32_t>();
    data[0] = origImg.rows;
    data[1] = origImg.cols;
    request.set_tensor(inputsNames[1], info);
    return ImageModel::preprocess(inputData, request);
}

namespace {
float getScore(const ov::Tensor& scoresTensor, size_t classInd, size_t boxInd) {
    return scoresTensor.data<float>()[classInd * scoresTensor.get_shape()[2] + boxInd];
}
}

std::unique_ptr<ResultBase> ModelYoloV3ONNX::postprocess(InferenceResult& infResult) {
    // Get info about input image
    const auto imgWidth = infResult.internalModelData->asRef<InternalImageModelData>().inputImgWidth;
    const auto imgHeight = infResult.internalModelData->asRef<InternalImageModelData>().inputImgHeight;

    // Get outputs tensors
    const ov::Tensor& boxes = infResult.outputsData[boxesOutputName];
    const float* boxesPtr = boxes.data<float>();

    const ov::Tensor& scores = infResult.outputsData[scoresOutputName];
    const ov::Tensor& indices = infResult.outputsData[indicesOutputName];

    const int* indicesData = indices.data<int>();
    const auto indicesShape = indices.get_shape();
    const auto boxShape = boxes.get_shape();

    // Generate detection results
    DetectionResult* result = new DetectionResult(infResult.frameId, infResult.metaData);
    size_t numberOfBoxes = indicesShape.size() == 3 ? indicesShape[1] : indicesShape[0];
    size_t indicesStride = indicesShape.size() == 3 ? indicesShape[2] : indicesShape[1];

    for (size_t i = 0; i < numberOfBoxes; ++i) {
        int batchInd = indicesData[i * indicesStride];
        int classInd = indicesData[i * indicesStride + 1];
        int boxInd = indicesData[i * indicesStride + 2];

        if (batchInd == -1) {
            break;
        }

        float score = getScore(scores, classInd, boxInd);

        if (score > confidenceThreshold) {
            DetectedObject obj;
            size_t startPos = boxShape[2] * boxInd;

            auto x = boxesPtr[startPos + 1];
            auto y = boxesPtr[startPos];
            auto width = boxesPtr[startPos + 3] - x;
            auto height = boxesPtr[startPos + 2] - y;

            // Create new detected box
            obj.x = clamp(x, 0.f, static_cast<float>(imgWidth));
            obj.y = clamp(y, 0.f, static_cast<float>(imgHeight));
            obj.height = clamp(height, 0.f, static_cast<float>(imgHeight));
            obj.width = clamp(width, 0.f, static_cast<float>(imgWidth));
            obj.confidence = score;
            obj.labelID = classInd;
            obj.label = getLabelName(classInd);

            result->objects.push_back(obj);

        }
    }

    return std::unique_ptr<ResultBase>(result);
}
