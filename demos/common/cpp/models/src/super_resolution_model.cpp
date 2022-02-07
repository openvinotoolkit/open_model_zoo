/*
// Copyright (C) 2021 Intel Corporation
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
#include <utils/image_utils.h>
#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>

#include "models/super_resolution_model.h"

SuperResolutionModel::SuperResolutionModel(const std::string& modelFileName, const cv::Size& inputImgSize) :
    ImageModel(modelFileName) {
        netInputHeight = inputImgSize.height;
        netInputWidth = inputImgSize.width;
}

void SuperResolutionModel::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output ---------------------------------------------
    // --------------------------- Prepare input --------------------------------------------------

    const ov::OutputVector& inputsInfo = model->inputs();
    if (inputsInfo.size() != 1 && inputsInfo.size() != 2) {
        throw std::logic_error("Super resolution model wrapper supports topologies with 1 or 2 inputs only");
    }
    std::string lrInputTensorName = inputsInfo.begin()->get_any_name();
    inputsNames.push_back(lrInputTensorName);
    ov::Shape lrShape = inputsInfo.begin()->get_shape();
    if (lrShape.size() != 4) {
        throw std::logic_error("Number of dimensions for an input must be 4");
    }

    ov::Layout inputLayout = ov::layout::get_layout(model->inputs().front());
    if (inputLayout.empty()) {
        inputLayout = { "NCHW" };
    }
    auto channelsId = ov::layout::channels_idx(inputLayout);
    auto heightId = ov::layout::height_idx(inputLayout);
    auto widthId = ov::layout::width_idx(inputLayout);

    if (lrShape[channelsId] != 1 && lrShape[channelsId] != 3) {
        throw std::logic_error("Input layer is expected to have 1 or 3 channels");
    }

    // A model like single-image-super-resolution-???? may take bicubic interpolation of the input image as the
    // second input
    std::string bicInputTensorName;
    if (inputsInfo.size() == 2) {
        bicInputTensorName = (++inputsInfo.begin())->get_any_name();
        inputsNames.push_back(bicInputTensorName);
        ov::Shape bicShape = (++inputsInfo.begin())->get_shape();
        if (bicShape.size() != 4) {
            throw std::logic_error("Number of dimensions for both inputs must be 4");
        }
        if (lrShape[widthId] >= bicShape[widthId] && lrShape[heightId] >= bicShape[heightId]) {
            std::swap(bicShape, lrShape);
            inputsNames[0].swap(inputsNames[1]);
        } else if (!(lrShape[widthId] <= bicShape[widthId] && lrShape[heightId] <= bicShape[heightId])) {
            throw std::logic_error("Each spatial dimension of one input must surpass or be equal to a spatial"
                "dimension of another input");
        }
    }

    ov::preprocess::PrePostProcessor ppp(model);
    for (const auto& input : inputsInfo) {
        ppp.input(input.get_any_name()).tensor().
            set_element_type(ov::element::u8).
            set_layout("NHWC");

        ppp.input(input.get_any_name()).model().set_layout("NCHW");
    }

    // --------------------------- Prepare output -----------------------------------------------------
    const ov::OutputVector& outputsInfo = model->outputs();
    if (outputsInfo.size() != 1) {
        throw std::logic_error("Super resolution model wrapper supports topologies only with 1 output");
    }

    outputsNames.push_back(outputsInfo.begin()->get_any_name());
    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();

    const ov::Shape& outShape = model->output().get_shape();

    ov::Layout outputLayout("NCHW");
    auto outWidth = outShape[ov::layout::width_idx(outputLayout)];
    auto inWidth = lrShape[ov::layout::width_idx(outputLayout)];
    changeInputSize(model, static_cast<int>(outWidth / inWidth));
}

void SuperResolutionModel::changeInputSize(std::shared_ptr<ov::Model>& model, int coeff) {
    std::map<std::string, ov::PartialShape> shapes;
    ov::Layout layout = ov::layout::get_layout(model->inputs().front());
    auto batchId = ov::layout::batch_idx(layout);
    auto heightId = ov::layout::height_idx(layout);
    auto widthId = ov::layout::width_idx(layout);

    const ov::OutputVector& inputsInfo = model->inputs();
    std::string lrInputTensorName = inputsInfo.begin()->get_any_name();
    ov::Shape lrShape = inputsInfo.begin()->get_shape();

    if (inputsInfo.size() == 2) {
        std::string bicInputTensorName = (++inputsInfo.begin())->get_any_name();
        ov::Shape bicShape = (++inputsInfo.begin())->get_shape();
        if (lrShape[heightId] >= bicShape[heightId] && lrShape[widthId] >= bicShape[widthId]) {
            std::swap(bicShape, lrShape);
            std::swap(bicInputTensorName, lrInputTensorName);
        }
        bicShape[batchId] = 1;
        bicShape[heightId] = coeff * netInputHeight;
        bicShape[widthId] = coeff * netInputWidth;
        shapes[bicInputTensorName] = ov::PartialShape(bicShape);
    }

    lrShape[batchId] = 1;
    lrShape[heightId] = netInputHeight;
    lrShape[widthId] = netInputWidth;
    shapes[lrInputTensorName] = ov::PartialShape(lrShape);

    model->reshape(shapes);
}

std::shared_ptr<InternalModelData> SuperResolutionModel::preprocess(const InputData& inputData, ov::InferRequest& request) {
    auto imgData = inputData.asRef<ImageInputData>();
    auto& img = imgData.inputImage;

    /* Resize and copy data from the image to the input Tensor */
    ov::Tensor lrInputTensor = request.get_tensor(inputsNames[0]);
    ov::Layout layout("NHWC");

    if (img.channels() != (int)lrInputTensor.get_shape()[ov::layout::channels_idx(layout)]) {
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }

    if (static_cast<size_t>(img.cols) != netInputWidth || static_cast<size_t>(img.rows) != netInputHeight) {
        slog::warn << "\tChosen model aspect ratio doesn't match image aspect ratio" << slog::endl;
    }
    const size_t height = lrInputTensor.get_shape()[ov::layout::height_idx(layout)];
    const size_t width = lrInputTensor.get_shape()[ov::layout::width_idx(layout)];
    img = resizeImageExt(img, width, height);
    request.set_tensor(inputsNames[0], wrapMat2Tensor(img));

    if (inputsNames.size() == 2) {
        ov::Tensor bicInputTensor = request.get_tensor(inputsNames[1]);
        int h = (int)bicInputTensor.get_shape()[ov::layout::height_idx(layout)];
        int w = (int)bicInputTensor.get_shape()[ov::layout::width_idx(layout)];
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(w, h), 0, 0, cv::INTER_CUBIC);
        request.set_tensor(inputsNames[1], wrapMat2Tensor(resized));
    }

    return std::make_shared<InternalImageModelData>(img.cols, img.rows);
}

std::unique_ptr<ResultBase> SuperResolutionModel::postprocess(InferenceResult& infResult) {
    ImageResult* result = new ImageResult;
    *static_cast<ResultBase*>(result) = static_cast<ResultBase&>(infResult);
    const auto outputData = infResult.getFirstOutputTensor().data<float>();

    std::vector<cv::Mat> imgPlanes;
    const ov::Shape& outShape = infResult.getFirstOutputTensor().get_shape();
    size_t outChannels = (int)(outShape[1]);
    size_t outHeight = (int)(outShape[2]);
    size_t outWidth = (int)(outShape[3]);
    size_t numOfPixels = outWidth * outHeight;
    if (outChannels == 3) {
        imgPlanes = std::vector<cv::Mat>{
              cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[0])),
              cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[numOfPixels])),
              cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[numOfPixels * 2]))};
    } else {
        imgPlanes = std::vector<cv::Mat>{cv::Mat(outHeight, outWidth, CV_32FC1, &(outputData[0]))};
        // Post-processing for text-image-super-resolution models
        cv::threshold(imgPlanes[0], imgPlanes[0], 0.5f, 1.0f, cv::THRESH_BINARY);
    }
    for (auto & img : imgPlanes)
        img.convertTo(img, CV_8UC1, 255);

    cv::Mat resultImg;
    cv::merge(imgPlanes, resultImg);
    result->resultImage = resultImg;

    return std::unique_ptr<ResultBase>(result);
}
