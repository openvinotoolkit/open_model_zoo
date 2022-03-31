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

#include <stddef.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>

#include "models/image_model.h"
#include "models/input_data.h"
#include "models/internal_model_data.h"
#include "models/jpeg_restoration_model.h"
#include "models/results.h"

JPEGRestorationModel::JPEGRestorationModel(const std::string& modelFileName,
                                           const cv::Size& inputImgSize,
                                           bool _jpegCompression,
                                           const std::string& layout)
    : ImageModel(modelFileName, false, layout) {
    netInputHeight = inputImgSize.height;
    netInputWidth = inputImgSize.width;
    jpegCompression = _jpegCompression;
}

void JPEGRestorationModel::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input  ------------------------------------------------------
    if (model->inputs().size() != 1) {
        throw std::logic_error("The JPEG Restoration model wrapper supports topologies with only 1 input");
    }
    inputsNames.push_back(model->input().get_any_name());

    const ov::Shape& inputShape = model->input().get_shape();
    const ov::Layout& inputLayout = getInputLayout(model->input());

    if (inputShape.size() != 4 || inputShape[ov::layout::batch_idx(inputLayout)] != 1 ||
        inputShape[ov::layout::channels_idx(inputLayout)] != 3) {
        throw std::logic_error("3-channel 4-dimensional model's input is expected");
    }

    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC");

    ppp.input().model().set_layout(inputLayout);

    // --------------------------- Prepare output  -----------------------------------------------------
    const ov::OutputVector& outputs = model->outputs();
    if (outputs.size() != 1) {
        throw std::logic_error("The JPEG Restoration model wrapper supports topologies with only 1 output");
    }
    const ov::Shape& outputShape = model->output().get_shape();
    const ov::Layout outputLayout{"NCHW"};
    if (outputShape.size() != 4 || outputShape[ov::layout::batch_idx(outputLayout)] != 1 ||
        outputShape[ov::layout::channels_idx(outputLayout)] != 3) {
        throw std::logic_error("3-channel 4-dimensional model's output is expected");
    }

    outputsNames.push_back(model->output().get_any_name());
    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();

    changeInputSize(model);
}

void JPEGRestorationModel::changeInputSize(std::shared_ptr<ov::Model>& model) {
    ov::Shape inputShape = model->input().get_shape();
    const ov::Layout& layout = ov::layout::get_layout(model->input());

    const auto batchId = ov::layout::batch_idx(layout);
    const auto heightId = ov::layout::height_idx(layout);
    const auto widthId = ov::layout::width_idx(layout);

    if (inputShape[heightId] % stride || inputShape[widthId] % stride) {
        throw std::logic_error("The shape of the model input must be divisible by stride");
    }

    netInputHeight = static_cast<int>((netInputHeight + stride - 1) / stride) * stride;
    netInputWidth = static_cast<int>((netInputWidth + stride - 1) / stride) * stride;

    inputShape[batchId] = 1;
    inputShape[heightId] = netInputHeight;
    inputShape[widthId] = netInputWidth;

    model->reshape(inputShape);
}

std::shared_ptr<InternalModelData> JPEGRestorationModel::preprocess(const InputData& inputData,
                                                                    ov::InferRequest& request) {
    cv::Mat image = inputData.asRef<ImageInputData>().inputImage;
    const size_t h = image.rows;
    const size_t w = image.cols;
    cv::Mat resizedImage;
    if (jpegCompression) {
        std::vector<uchar> encimg;
        std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, 40};
        cv::imencode(".jpg", image, encimg, params);
        image = cv::imdecode(cv::Mat(encimg), 3);
    }

    if (netInputHeight - stride < h && h <= netInputHeight && netInputWidth - stride < w && w <= netInputWidth) {
        int bottom = netInputHeight - h;
        int right = netInputWidth - w;
        cv::copyMakeBorder(image, resizedImage, 0, bottom, 0, right, cv::BORDER_CONSTANT, 0);
    } else {
        slog::warn << "\tChosen model aspect ratio doesn't match image aspect ratio" << slog::endl;
        cv::resize(image, resizedImage, cv::Size(netInputWidth, netInputHeight));
    }
    request.set_input_tensor(wrapMat2Tensor(resizedImage));

    return std::make_shared<InternalImageModelData>(image.cols, image.rows);
}

std::unique_ptr<ResultBase> JPEGRestorationModel::postprocess(InferenceResult& infResult) {
    ImageResult* result = new ImageResult;
    *static_cast<ResultBase*>(result) = static_cast<ResultBase&>(infResult);

    const auto& inputImgSize = infResult.internalModelData->asRef<InternalImageModelData>();
    const auto outputData = infResult.getFirstOutputTensor().data<float>();

    std::vector<cv::Mat> imgPlanes;
    const ov::Shape& outputShape = infResult.getFirstOutputTensor().get_shape();
    const size_t outHeight = static_cast<int>(outputShape[2]);
    const size_t outWidth = static_cast<int>(outputShape[3]);
    const size_t numOfPixels = outWidth * outHeight;
    imgPlanes = std::vector<cv::Mat>{cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[0])),
                                     cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[numOfPixels])),
                                     cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[numOfPixels * 2]))};
    cv::Mat resultImg;
    cv::merge(imgPlanes, resultImg);

    if (netInputHeight - stride < static_cast<size_t>(inputImgSize.inputImgHeight) &&
        static_cast<size_t>(inputImgSize.inputImgHeight) <= netInputHeight &&
        netInputWidth - stride < static_cast<size_t>(inputImgSize.inputImgWidth) &&
        static_cast<size_t>(inputImgSize.inputImgWidth) <= netInputWidth) {
        result->resultImage = resultImg(cv::Rect(0, 0, inputImgSize.inputImgWidth, inputImgSize.inputImgHeight));
    } else {
        cv::resize(resultImg, result->resultImage, cv::Size(inputImgSize.inputImgWidth, inputImgSize.inputImgHeight));
    }

    result->resultImage.convertTo(result->resultImage, CV_8UC3, 255);

    return std::unique_ptr<ResultBase>(result);
}
