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
#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>

#include "models/super_resolution_model.h"

SuperResolutionModel::SuperResolutionModel(const std::string& modelFileName, const cv::Size& inputImgSize) :
    ImageModel(modelFileName, false) {
        netInputHeight = inputImgSize.height;
        netInputWidth = inputImgSize.width;
}

void SuperResolutionModel::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output ---------------------------------------------
    // --------------------------- Prepare input --------------------------------------------------

    const ov::OutputVector& inputsInfo = model->inputs();
    if (inputsInfo.size() != 1 && inputsInfo.size() != 2) {
        throw std::runtime_error("Super resolution model wrapper supports topologies with 1 or 2 inputs only");
    }
    std::string lrInputTensorName = inputsInfo.begin()->get_any_name();
    inputsNames.push_back(lrInputTensorName);
    const ov::Shape& lrShape = inputsInfo.begin()->get_shape();
    if (lrShape.size() != 4) {
        throw std::runtime_error("Number of dimensions for an input must be 4");
    }
    if (lrShape[1] != 1 && lrShape[1] != 3) {
        throw std::runtime_error("Input layer is expected to have 1 or 3 channels");
    }

    // A model like single-image-super-resolution-???? may take bicubic interpolation of the input image as the
    // second input
    std::string bicInputTensorName;
    if (inputsInfo.size() == 2) {
        bicInputTensorName = (++inputsInfo.begin())->get_any_name();
        inputsNames.push_back(bicInputTensorName);
        const ov::Shape& bicShape = (++inputsInfo.begin())->get_shape();
        if (bicShape.size() != 4) {
            throw std::runtime_error("Number of dimensions for both inputs must be 4");
        }
        if (lrShape[2] >= bicShape[2] && lrShape[3] >= bicShape[3]) {
            inputsNames[0].swap(inputsNames[1]);
        } else if (!(lrShape[2] <= bicShape[2] && lrShape[3] <= bicShape[3])) {
            throw std::runtime_error("Each spatial dimension of one input must surpass or be equal to a spatial"
                "dimension of another input");
        }
    }

    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor().
        set_element_type(ov::element::f32).
        set_layout({ "NHWC" });

    ppp.input().model().set_layout("NCHW");

    //inputInfo.setPrecision(InferenceEngine::Precision::FP32);
    // --------------------------- Prepare output -----------------------------------------------------
    const ov::OutputVector& outputsInfo = model->outputs();
    if (outputsInfo.size() != 1) {
        throw std::runtime_error("Super resolution model wrapper supports topologies only with 1 output");
    }

    outputsNames.push_back(outputsInfo.begin()->get_any_name());
    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();

    const ov::Shape& outShape = model->output().get_shape();
    changeInputSize(model, outShape[2] / lrShape[2]);
}

void SuperResolutionModel::changeInputSize(std::shared_ptr<ov::Model>& model, int coeff) {
    std::map<std::string, ov::PartialShape> shapes;
    const ov::OutputVector& inputsInfo = model->inputs();
    std::string lrInputTensorName = inputsInfo.begin()->get_any_name();
    ov::Shape lrShape = inputsInfo.begin()->get_shape();

    lrShape[0] = 1;
    lrShape[2] = netInputHeight;
    lrShape[3] = netInputWidth;
    shapes[lrInputTensorName] = ov::PartialShape(lrShape);

    if (inputsInfo.size() == 2) {
        std::string bicInputTensorName = inputsInfo.begin()->get_any_name();
        ov::Shape bicShape = (++inputsInfo.begin())->get_shape();
        bicShape[0] = 1;
        bicShape[2] = coeff * netInputHeight;
        bicShape[3] = coeff * netInputWidth;
        shapes[bicInputTensorName] = ov::PartialShape(bicShape);
    }

    model->reshape(shapes);
}

std::shared_ptr<InternalModelData> SuperResolutionModel::preprocess(const InputData& inputData, ov::runtime::InferRequest& request) {
    auto imgData = inputData.asRef<ImageInputData>();
    auto& img = imgData.inputImage;

    /* Resize and copy data from the image to the input Tensor */
    ov::Tensor lrInputTensor = request.get_tensor(inputsNames[0]);
    if (img.channels() != (int)lrInputTensor.get_shape()[1]) {
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }

    if (static_cast<size_t>(img.cols) != netInputWidth || static_cast<size_t>(img.rows) != netInputHeight) {
        slog::warn << "\tChosen model aspect ratio doesn't match image aspect ratio" << slog::endl;
    }
    matToTensor(img, lrInputTensor);

    if (inputsNames.size() == 2) {
        ov::Tensor bicInputTensor = request.get_tensor(inputsNames[1]);

        int w = bicInputTensor.get_shape()[3];
        int h = bicInputTensor.get_shape()[2];
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(w, h), 0, 0, cv::INTER_CUBIC);
        matToTensor(resized, bicInputTensor);
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
