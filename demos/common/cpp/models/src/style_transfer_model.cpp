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

#include "models/style_transfer_model.h"

#include <stddef.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include <utils/image_utils.h>
#include <utils/ocv_common.hpp>

#include "models/input_data.h"
#include "models/internal_model_data.h"
#include "models/results.h"

StyleTransferModel::StyleTransferModel(const std::string& modelFileName, const std::string& layout)
    : ImageModel(modelFileName, false, layout) {}

void StyleTransferModel::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output ---------------------------------------------
    // --------------------------- Prepare input --------------------------------------------------
    if (model->inputs().size() != 1) {
        throw std::logic_error("Style transfer model wrapper supports topologies with only 1 input");
    }

    inputsNames.push_back(model->input().get_any_name());

    const ov::Shape& inputShape = model->input().get_shape();
    ov::Layout inputLayout = getInputLayout(model->input());

    if (inputShape.size() != 4 || inputShape[ov::layout::batch_idx(inputLayout)] != 1 ||
        inputShape[ov::layout::channels_idx(inputLayout)] != 3) {
        throw std::logic_error("3-channel 4-dimensional model's input is expected");
    }

    netInputWidth = inputShape[ov::layout::width_idx(inputLayout)];
    netInputHeight = inputShape[ov::layout::height_idx(inputLayout)];

    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().preprocess().convert_element_type(ov::element::f32);
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC");

    ppp.input().model().set_layout(inputLayout);

    // --------------------------- Prepare output  -----------------------------------------------------
    const ov::OutputVector& outputs = model->outputs();
    if (outputs.size() != 1) {
        throw std::logic_error("Style transfer model wrapper supports topologies with only 1 output");
    }
    outputsNames.push_back(model->output().get_any_name());

    const ov::Shape& outputShape = model->output().get_shape();
    ov::Layout outputLayout{"NCHW"};
    if (outputShape.size() != 4 || outputShape[ov::layout::batch_idx(outputLayout)] != 1 ||
        outputShape[ov::layout::channels_idx(outputLayout)] != 3) {
        throw std::logic_error("3-channel 4-dimensional model's output is expected");
    }

    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();
}

std::shared_ptr<InternalModelData> StyleTransferModel::preprocess(const InputData& inputData,
                                                                  ov::InferRequest& request) {
    auto imgData = inputData.asRef<ImageInputData>();
    auto& img = imgData.inputImage;

    cv::Mat resizedImage;
    resizedImage = resizeImageExt(img, netInputWidth, netInputHeight);
    request.set_input_tensor(wrapMat2Tensor(resizedImage));
    return std::make_shared<InternalImageModelData>(img.cols, img.rows);
}

std::unique_ptr<ResultBase> StyleTransferModel::postprocess(InferenceResult& infResult) {
    ImageResult* result = new ImageResult;
    *static_cast<ResultBase*>(result) = static_cast<ResultBase&>(infResult);

    const auto& inputImgSize = infResult.internalModelData->asRef<InternalImageModelData>();
    const auto outputData = infResult.getFirstOutputTensor().data<float>();

    const ov::Shape& outputShape = infResult.getFirstOutputTensor().get_shape();
    size_t outHeight = static_cast<int>(outputShape[2]);
    size_t outWidth = static_cast<int>(outputShape[3]);
    size_t numOfPixels = outWidth * outHeight;

    std::vector<cv::Mat> imgPlanes;
    imgPlanes = std::vector<cv::Mat>{cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[numOfPixels * 2])),
                                     cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[numOfPixels])),
                                     cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[0]))};
    cv::Mat resultImg;
    cv::merge(imgPlanes, resultImg);
    cv::resize(resultImg, result->resultImage, cv::Size(inputImgSize.inputImgWidth, inputImgSize.inputImgHeight));

    result->resultImage.convertTo(result->resultImage, CV_8UC3);

    return std::unique_ptr<ResultBase>(result);
}
