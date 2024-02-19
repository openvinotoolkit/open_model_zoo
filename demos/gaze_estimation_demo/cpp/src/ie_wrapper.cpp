// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"

#include <utils/common.hpp>

#include "ie_wrapper.hpp"

namespace gaze_estimation {

IEWrapper::IEWrapper(
    ov::Core& core, const std::string& modelPath, const std::string& modelType, const std::string& deviceName) :
        modelPath(modelPath), modelType(modelType), deviceName(deviceName), core(core)
{
    slog::info << "Reading model: " << modelPath << slog::endl;
    model = core.read_model(modelPath);
    logBasicModelInfo(model);
    setExecPart();
}

void IEWrapper::setExecPart() {
    // set map of input tensor name -- tensor dimension pairs
    ov::OutputVector inputs = model->inputs();
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
    for (size_t i = 0; i < inputs.size(); i++) {
        std::string layerName = inputs[i].get_any_name();
        ov::Shape layerDims = inputs[i].get_shape();
        input_tensors_dims_info[layerName] = layerDims;
        if (layerDims.size() == 4) {
            ppp.input(layerName).tensor().
                set_element_type(ov::element::u8).
                set_layout({ "NCHW" });
        }
        else if (layerDims.size() == 2) {
            ppp.input(layerName).tensor().
                set_element_type(ov::element::f32).
                set_layout({ "NC" });
        }
        else {
            throw std::runtime_error("Unknown type of input layer layout. Expected either 4 or 2 dimensional inputs");
        }

    }
    // set map of output tensor name -- tensor dimension pairs
    ov::OutputVector outputs = model->outputs();
    for (size_t i = 0; i < outputs.size(); i++) {
        std::string layerName = outputs[i].get_any_name();
        ov::Shape layerDims = outputs[i].get_shape();
        output_tensors_dims_info[layerName] = layerDims;
        ppp.output(layerName).tensor().set_element_type(ov::element::f32);
    }
    model = ppp.build();

    compiled_model = core.compile_model(model, deviceName);
    logCompiledModelInfo(compiled_model, modelPath, deviceName);
    infer_request = compiled_model.create_infer_request();
}

void IEWrapper::setInputTensor(const std::string& tensorName, const cv::Mat& image) {
    auto tensorDims = input_tensors_dims_info[tensorName];

    if (tensorDims.size() != 4) {
        throw std::runtime_error("Input data does not match size of the tensor");
    }

    auto scaledSize = cv::Size(static_cast<int>(tensorDims[3]), static_cast<int>(tensorDims[2]));
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, scaledSize, 0, 0, cv::INTER_CUBIC);

    ov::Tensor input_tensor = infer_request.get_tensor(tensorName);
    matToTensor(resizedImage, input_tensor);
}

void IEWrapper::setInputTensor(const std::string& tensorName, const std::vector<float>& data) {
    auto tensorDims = input_tensors_dims_info[tensorName];
    size_t dimsProduct = std::accumulate(tensorDims.begin(), tensorDims.end(), 1, std::multiplies<size_t>());
    if (dimsProduct != data.size()) {
        throw std::runtime_error("Input data does not match size of the tensor");
    }

    float* buffer = infer_request.get_tensor(tensorName).data<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        buffer[i] = data[i];
    }
}

void IEWrapper::getOutputTensor(const std::string& tensorName, std::vector<float>& output) {
    output.clear();
    auto tensorDims = output_tensors_dims_info[tensorName];
    size_t dataSize = std::accumulate(tensorDims.begin(), tensorDims.end(), 1, std::multiplies<size_t>());
    float* buffer = infer_request.get_tensor(tensorName).data<float>();

    for (size_t i = 0; i < dataSize; ++i) {
        output.push_back(buffer[i]);
    }
}

const std::map<std::string, ov::Shape>& IEWrapper::getInputTensorDimsInfo() const {
    return input_tensors_dims_info;
}
const std::map<std::string, ov::Shape>& IEWrapper::getOutputTensorDimsInfo() const {
    return output_tensors_dims_info;
}

std::string IEWrapper::expectSingleInput() const {
    if (input_tensors_dims_info.size() != 1) {
        throw std::runtime_error(modelPath + ": expected to have 1 input");
    }

    return input_tensors_dims_info.begin()->first;
}

std::string IEWrapper::expectSingleOutput() const {
    if (output_tensors_dims_info.size() != 1) {
        throw std::runtime_error(modelPath + ": expected to have 1 output");
    }

    return output_tensors_dims_info.begin()->first;
}

void IEWrapper::expectImageInput(const std::string& tensorName) const {
    const auto& dims = input_tensors_dims_info.at(tensorName);

    if (dims.size() != 4 || dims[0] != 1 || dims[1] != 3) {
        throw std::runtime_error(modelPath + ": expected \"" + tensorName + "\" to have dimensions 1x3xHxW");
    }
}

void IEWrapper::infer() {
    infer_request.infer();
}

void IEWrapper::reshape(const std::map<std::string, ov::Shape>& newTensorsDimsInfo) {
    if (input_tensors_dims_info.size() != newTensorsDimsInfo.size()) {
        throw std::runtime_error("Mismatch in the number of tensors being reshaped");
    }

    std::map<std::string, ov::PartialShape> partial_shapes;
    for (const std::pair<std::string, ov::Shape>& it : newTensorsDimsInfo) {
        ov::PartialShape newShape(it.second);
        partial_shapes[it.first] = newShape;
    }
    model->reshape(partial_shapes);
    setExecPart();
}
}  // namespace gaze_estimation
