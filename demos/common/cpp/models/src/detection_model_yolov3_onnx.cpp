/*
// Copyright (C) 2022 Intel Corporation
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
        interpolationMode = CUBIC;
        resizeMode = RESIZE_KEEP_ASPECT_LETTERBOX;
    }


void ModelYoloV3ONNX::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input  ------------------------------------------------------
    const ov::OutputVector& inputs = model->inputs();
    if (inputs.size() != 2) {
        throw std::logic_error("YoloV3ONNX model wrapper expects models that have 2 inputs");
    }

    // Check first image input
    std::string imageInputName = inputs.begin()->get_any_name();
    inputsNames.push_back(imageInputName);

    const ov::Shape& imageShape = inputs.begin()->get_shape();
    const ov::Layout& imageLayout = getInputLayout(inputs.front());

    if (imageShape.size() != 4 && imageShape[ov::layout::channels_idx(imageLayout)] != 3) {
        throw std::logic_error("Expected 4D image input with 3 channels");
    }

    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input(imageInputName).tensor().set_element_type(ov::element::u8).set_layout({"NHWC"});

    ppp.input(imageInputName).model().set_layout(imageLayout);

    // Check second info input
    std::string infoInputName = (++inputs.begin())->get_any_name();
    inputsNames.push_back(infoInputName);

    const ov::Shape infoShape = (++inputs.begin())->get_shape();
    const ov::Layout& infoLayout = getInputLayout(inputs.at(1));

    if (infoShape.size() != 2 && infoShape[ov::layout::channels_idx(infoLayout)] != 2) {
            throw std::logic_error("Expected 2D image info input with 2 channels");
        }

    ppp.input(infoInputName).tensor().set_element_type(ov::element::i32);

    ppp.input(infoInputName).model().set_layout(infoLayout);

    // --------------------------- Reading image input parameters -------------------------------------------
    netInputWidth = imageShape[ov::layout::width_idx(imageLayout)];
    netInputHeight = imageShape[ov::layout::height_idx(imageLayout)];

    // --------------------------- Prepare output  -----------------------------------------------------
    if (model->outputs().size() != 3) {
        throw std::logic_error("YoloV3ONNX model wrapper expects models that have 3 outputs");
    }

    const ov::OutputVector& outputs = model->outputs();
    for (auto& output : outputs) {
        const ov::Shape& currentShape = output.get_partial_shape().get_max_shape();
        std::string currentName = output.get_any_name();
        if (currentShape[currentShape.size() - 1] == 3) {
            indicesOuputName = currentName;
            ppp.output(currentName).tensor().set_element_type(ov::element::i32);
        } else if (currentShape[2] == 4) {
            boxesOutputName = currentName;
            ppp.output(currentName).tensor().set_element_type(ov::element::f32);
        } else if (currentShape[1] == numberOfClasses) {
            scoresOutputName = currentName;
            ppp.output(currentName).tensor().set_element_type(ov::element::f32);
        } else {
            throw std::logic_error("Expected shapes [:,:,4], [:,numClasses,:] and [:,3] for outputs");
        }
        outputsNames.push_back(currentName);
    }
    model = ppp.build();
}

std::shared_ptr<InternalModelData> ModelYoloV3ONNX::preprocess(const InputData& inputData,
                                                               ov::InferRequest& request) {
    const auto& origImg = inputData.asRef<ImageInputData>().inputImage;

    int* img_size = new int[2];
    img_size[0] = origImg.rows;
    img_size[1] = origImg.cols;
    ov::Tensor infoInput = ov::Tensor(ov::element::i32, ov::Shape({1, 2}), img_size);

    request.set_tensor(inputsNames[1], infoInput);

    return ImageModel::preprocess(inputData, request);
}

float ModelYoloV3ONNX::getScore(const ov::Tensor& scoresTensor, size_t classInd, size_t boxInd) {
    const float* scoresPtr = scoresTensor.data<float>();
    const auto shape = scoresTensor.get_shape();
    int N = shape[2];

    return scoresPtr[classInd * N + boxInd];
}

std::unique_ptr<ResultBase> ModelYoloV3ONNX::postprocess(InferenceResult& infResult) {
    // Get info about input image
    const auto imgWidth = infResult.internalModelData->asRef<InternalImageModelData>().inputImgWidth;
    const auto imgHeight = infResult.internalModelData->asRef<InternalImageModelData>().inputImgHeight;

    // Get outputs tensors
    const ov::Tensor& boxes = infResult.outputsData[boxesOutputName];
    const float* boxesPtr = boxes.data<float>();

    const ov::Tensor& scores = infResult.outputsData[scoresOutputName];
    const ov::Tensor& indices = infResult.outputsData[indicesOuputName];

    const int* indicesData = indices.data<int>();
    const auto indicesShape = indices.get_shape();
    const auto boxShape = boxes.get_shape();

    // Generate detection results
    DetectionResult* result = new DetectionResult(infResult.frameId, infResult.metaData);
    size_t numberOfBoxes = indicesShape.size() == 3 ? indicesShape[1] : indicesShape[0];
    int indicesStride = indicesShape.size() == 3 ? indicesShape[2] : indicesShape[1];

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
