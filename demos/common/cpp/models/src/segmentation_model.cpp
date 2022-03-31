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

#include "models/segmentation_model.h"

#include <stddef.h>
#include <stdint.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include "models/internal_model_data.h"
#include "models/results.h"

SegmentationModel::SegmentationModel(const std::string& modelFileName, bool useAutoResize, const std::string& layout)
    : ImageModel(modelFileName, useAutoResize, layout) {}

std::vector<std::string> SegmentationModel::loadLabels(const std::string& labelFilename) {
    std::vector<std::string> labelsList;

    /* Read labels (if any) */
    if (!labelFilename.empty()) {
        std::ifstream inputFile(labelFilename);
        if (!inputFile.is_open())
            throw std::runtime_error("Can't open the labels file: " + labelFilename);
        std::string label;
        while (std::getline(inputFile, label)) {
            labelsList.push_back(label);
        }
        if (labelsList.empty())
            throw std::logic_error("File is empty: " + labelFilename);
    }

    return labelsList;
}

void SegmentationModel::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output ---------------------------------------------
    // --------------------------- Prepare input  -----------------------------------------------------
    if (model->inputs().size() != 1) {
        throw std::logic_error("Segmentation model wrapper supports topologies with only 1 input");
    }
    const auto& input = model->input();
    inputsNames.push_back(input.get_any_name());

    const ov::Layout& inputLayout = getInputLayout(input);
    const ov::Shape& inputShape = input.get_shape();
    if (inputShape.size() != 4 || inputShape[ov::layout::channels_idx(inputLayout)] != 3) {
        throw std::logic_error("3-channel 4-dimensional model's input is expected");
    }

    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout({"NHWC"});

    if (useAutoResize) {
        ppp.input().tensor().set_spatial_dynamic_shape();

        ppp.input()
            .preprocess()
            .convert_element_type(ov::element::f32)
            .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
    }

    ppp.input().model().set_layout(inputLayout);
    model = ppp.build();
    // --------------------------- Prepare output  -----------------------------------------------------
    if (model->outputs().size() != 1) {
        throw std::logic_error("Segmentation model wrapper supports topologies with only 1 output");
    }

    const auto& output = model->output();
    outputsNames.push_back(output.get_any_name());

    const ov::Shape& outputShape = output.get_shape();
    ov::Layout outputLayout("");
    switch (outputShape.size()) {
        case 3:
            outputLayout = "CHW";
            outChannels = 1;
            outHeight = static_cast<int>(outputShape[ov::layout::height_idx(outputLayout)]);
            outWidth = static_cast<int>(outputShape[ov::layout::width_idx(outputLayout)]);
            break;
        case 4:
            outputLayout = "NCHW";
            outChannels = static_cast<int>(outputShape[ov::layout::channels_idx(outputLayout)]);
            outHeight = static_cast<int>(outputShape[ov::layout::height_idx(outputLayout)]);
            outWidth = static_cast<int>(outputShape[ov::layout::width_idx(outputLayout)]);
            break;
        default:
            throw std::logic_error("Unexpected output tensor shape. Only 4D and 3D outputs are supported.");
    }
}

std::unique_ptr<ResultBase> SegmentationModel::postprocess(InferenceResult& infResult) {
    ImageResult* result = new ImageResult(infResult.frameId, infResult.metaData);
    const auto& inputImgSize = infResult.internalModelData->asRef<InternalImageModelData>();
    const auto& outTensor = infResult.getFirstOutputTensor();

    result->resultImage = cv::Mat(outHeight, outWidth, CV_8UC1);

    if (outChannels == 1 && outTensor.get_element_type() == ov::element::i32) {
        cv::Mat predictions(outHeight, outWidth, CV_32SC1, outTensor.data<int32_t>());
        predictions.convertTo(result->resultImage, CV_8UC1);
    } else if (outChannels == 1 && outTensor.get_element_type() == ov::element::i64) {
        cv::Mat predictions(outHeight, outWidth, CV_32SC1);
        const auto data = outTensor.data<int64_t>();
        for (size_t i = 0; i < predictions.total(); ++i) {
            reinterpret_cast<int32_t*>(predictions.data)[i] = int32_t(data[i]);
        }
        predictions.convertTo(result->resultImage, CV_8UC1);
    } else if (outTensor.get_element_type() == ov::element::f32) {
        const float* data = outTensor.data<float>();
        for (int rowId = 0; rowId < outHeight; ++rowId) {
            for (int colId = 0; colId < outWidth; ++colId) {
                int classId = 0;
                float maxProb = -1.0f;
                for (int chId = 0; chId < outChannels; ++chId) {
                    float prob = data[chId * outHeight * outWidth + rowId * outWidth + colId];
                    if (prob > maxProb) {
                        classId = chId;
                        maxProb = prob;
                    }
                }  // nChannels

                result->resultImage.at<uint8_t>(rowId, colId) = classId;
            }  // width
        }  // height
    }

    cv::resize(result->resultImage,
               result->resultImage,
               cv::Size(inputImgSize.inputImgWidth, inputImgSize.inputImgHeight),
               0,
               0,
               cv::INTER_NEAREST);

    return std::unique_ptr<ResultBase>(result);
}
