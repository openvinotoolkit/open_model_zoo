// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "openvino/openvino.hpp"

#include "utils/common.hpp"
#include "utils/ocv_common.hpp"

using namespace ov::preprocess;

class Detector {
public:
    struct Result {
        std::size_t label;
        float confidence;
        cv::Rect location;
    };

    static constexpr int maxProposalCount = 200;
    static constexpr int objectSize = 7;  // Output should have 7 as a last dimension"

    Detector() = default;
    Detector(ov::Core& core, const std::string& deviceName, const std::string& xmlPath, const std::vector<float>& detectionTresholds,
            const bool autoResize, const std::map<std::string, std::string>& pluginConfig) :
        detectionTresholds{detectionTresholds}, m_core{core} {
        slog::info << "Reading model: " << xmlPath << slog::endl;
        std::shared_ptr<ov::Model> model = core.read_model(xmlPath);
        logBasicModelInfo(model);

        // Check model inputs and outputs

        ov::OutputVector inputs = model->inputs();
        if (inputs.size() != 1) {
            throw std::logic_error("Detector should have only one input");
        }

        detectorInputName = model->input().get_any_name();

        ov::OutputVector outputs = model->outputs();
        if (outputs.size() != 1) {
            throw std::logic_error("Vehicle Detection network should have only one output");
        }

        ov::Output<ov::Node> output = outputs[0];

        detectorOutputName = output.get_any_name();
        ov::Shape output_shape = output.get_shape();

        if (output_shape.size() != 4) {
            throw std::logic_error("Incorrect output dimensions for SSD");
        }

        if (maxProposalCount != output_shape[2]) {
            throw std::logic_error("unexpected ProposalCount");
        }
        if (objectSize != output_shape[3]) {
            throw std::logic_error("Output should have 7 as a last dimension");
        }

        PrePostProcessor ppp(model);

        InputInfo& inputInfo = ppp.input();

        InputTensorInfo& inputTensorInfo = inputInfo.tensor();
        inputTensorInfo.set_element_type(ov::element::u8);
        if (FLAGS_auto_resize) {
            inputTensorInfo.set_layout({ "NHWC" });
            inputTensorInfo.set_spatial_dynamic_shape();
        } else {
            inputTensorInfo.set_layout({ "NCHW" });
        }

        PreProcessSteps& preProcessSteps = inputInfo.preprocess();
        preProcessSteps.convert_element_type(ov::element::f32);
        if (FLAGS_auto_resize) {
            preProcessSteps.resize(ResizeAlgorithm::RESIZE_LINEAR);
        }

        InputModelInfo& inputModelInfo = inputInfo.model();
        inputModelInfo.set_layout({ "NCHW" });

        model = ppp.build();

        m_compiled_model = m_core.compile_model(model, deviceName, pluginConfig);
        logCompiledModelInfo(m_compiled_model, xmlPath, deviceName, "Vehicle And License Plate Detection");
    }

    ov::InferRequest createInferRequest() {
        return m_compiled_model.create_infer_request();
    }

    void setImage(ov::InferRequest& inferRequest, const cv::Mat& img) {
        ov::Tensor inputTensor = inferRequest.get_tensor(detectorInputName);
        ov::Shape shape = inputTensor.get_shape();
        if (3 == shape[ov::layout::channels_idx(ov::Layout({ "NHWC" }))]) {
            // autoResize is set
            if (!img.isSubmatrix()) {
                // just wrap Mat object with Tensor without additional memory allocation
                ov::Tensor frameTensor = wrapMat2Tensor(img);
                inferRequest.set_tensor(detectorInputName, frameTensor);
            } else {
                throw std::logic_error("Sparse matrix are not supported");
            }
        } else {
            // resize and copy data from image to tensor using OpenCV
            matToTensor(img, inputTensor);
        }
    }

    std::list<Result> getResults(ov::InferRequest& inferRequest, cv::Size upscale, std::vector<std::string>& rawResults) {
        // there is no big difference if InferReq of detector from another device is passed
        // because the processing is the same for the same topology
        std::list<Result> results;
        ov::Tensor output_tensor = inferRequest.get_tensor(detectorOutputName);
        const float* const detections = output_tensor.data<float>();
        // pretty much regular SSD post-processing
        for (int i = 0; i < maxProposalCount; i++) {
            float image_id = detections[i * objectSize + 0]; // in case of batch
            if (image_id < 0) { // indicates end of detections
                break;
            }
            size_t label = static_cast<decltype(detectionTresholds.size())>(detections[i * objectSize + 1]);
            float confidence = detections[i * objectSize + 2];
            if (label - 1 < detectionTresholds.size() && confidence < detectionTresholds[label - 1]) {
                continue;
            }

            cv::Rect rect;
            rect.x = static_cast<int>(detections[i * objectSize + 3] * upscale.width);
            rect.y = static_cast<int>(detections[i * objectSize + 4] * upscale.height);
            rect.width = static_cast<int>(detections[i * objectSize + 5] * upscale.width) - rect.x;
            rect.height = static_cast<int>(detections[i * objectSize + 6] * upscale.height) - rect.y;
            results.push_back(Result{label, confidence, rect});
            std::ostringstream rawResultsStream;
            rawResultsStream << "[" << i << "," << label << "] element, prob = " << confidence
                        << "    (" << rect.x << "," << rect.y << ")-(" << rect.width << "," << rect.height << ")";
            rawResults.push_back(rawResultsStream.str());
        }
        return results;
    }

private:
    std::vector<float> detectionTresholds;
    std::string detectorInputName;
    std::string detectorOutputName;
    ov::Core m_core; // The only reason to store a plugin as to assure that it lives at least as long as ExecutableNetwork
    ov::CompiledModel m_compiled_model;
};

class VehicleAttributesClassifier {
public:
    VehicleAttributesClassifier() = default;
    VehicleAttributesClassifier(ov::Core& core, const std::string& deviceName,
        const std::string& xmlPath, const bool autoResize, const std::map<std::string, std::string>& pluginConfig) : m_core(core) {
        slog::info << "Reading model: " << xmlPath << slog::endl;
        std::shared_ptr<ov::Model> model = m_core.read_model(xmlPath);
        logBasicModelInfo(model);

        ov::OutputVector inputs = model->inputs();
        if (inputs.size() != 1) {
            throw std::logic_error("Vehicle Attribs topology should have only one input");
        }

        attributesInputName = model->input().get_any_name();

        ov::OutputVector outputs = model->outputs();
        if (outputs.size() != 2) {
            throw std::logic_error("Vehicle Attribs Network expects networks having two outputs");
        }

        // color is the first output
        outputNameForColor = outputs[0].get_any_name();
        // type is the second output.
        outputNameForType = outputs[1].get_any_name();

        PrePostProcessor ppp(model);

        InputInfo& inputInfo = ppp.input();
        InputTensorInfo& inputTensorInfo = inputInfo.tensor();
        inputTensorInfo.set_element_type(ov::element::u8);
        if (FLAGS_auto_resize) {
            inputTensorInfo.set_layout({ "NHWC" });
        } else {
            inputTensorInfo.set_layout({ "NCHW" });
        }

        PreProcessSteps& preProcessSteps = inputInfo.preprocess();
        preProcessSteps.convert_element_type(ov::element::f32);
        if (FLAGS_auto_resize) {
            preProcessSteps.resize(ResizeAlgorithm::RESIZE_LINEAR);
        }

        InputModelInfo& inputModelInfo = inputInfo.model();
        inputModelInfo.set_layout({ "NCHW" });

        model = ppp.build();

        m_compiled_model = m_core.compile_model(model, deviceName, pluginConfig);
        logCompiledModelInfo(m_compiled_model, FLAGS_m_va, deviceName, "Vehicle Attributes Recognition");
    }

    ov::InferRequest createInferRequest() {
        return m_compiled_model.create_infer_request();
    }

    void setImage(ov::InferRequest& inferRequest, const cv::Mat& img, const cv::Rect vehicleRect) {
        ov::Tensor inputTensor = inferRequest.get_tensor(attributesInputName);
        ov::Shape shape = inputTensor.get_shape();
        if (3 == shape[ov::layout::channels_idx(ov::Layout({ "NHWC" }))]) {
            // autoResize is set
            ov::Tensor frameTensor = wrapMat2Tensor(img);
            ov::Coordinate p00({ 0, (size_t)vehicleRect.y, (size_t)vehicleRect.x, 0 });
            ov::Coordinate p01({ 1, (size_t)(vehicleRect.y + vehicleRect.height), (size_t)vehicleRect.x + vehicleRect.width, 3 });
            ov::Tensor roiTensor(frameTensor, p00, p01);

            inferRequest.set_tensor(attributesInputName, roiTensor);
        } else {
            const cv::Mat& vehicleImage = img(vehicleRect);
            matToTensor(vehicleImage, inputTensor);
        }
    }

    std::pair<std::string, std::string> getResults(ov::InferRequest& inferRequest) {
        static const std::string colors[] = {
            "white", "gray", "yellow", "red", "green", "blue", "black"
        };
        static const std::string types[] = {
            "car", "van", "truck", "bus"
        };

        // 7 possible colors for each vehicle and we should select the one with the maximum probability
        ov::Tensor colorsTensor = inferRequest.get_tensor(outputNameForColor);
        const float* colorsValues = colorsTensor.data<float>();

        // 4 possible types for each vehicle and we should select the one with the maximum probability
        ov::Tensor typesTensor = inferRequest.get_tensor(outputNameForType);
        const float* typesValues = typesTensor.data<float>();

        const auto color_id = std::max_element(colorsValues, colorsValues + 7) - colorsValues;
        const auto  type_id = std::max_element(typesValues,  typesValues  + 4) - typesValues;

        return std::pair<std::string, std::string>(colors[color_id], types[type_id]);
    }

private:
    std::string attributesInputName;
    std::string outputNameForColor;
    std::string outputNameForType;
    ov::Core m_core;  // The only reason to store a device is to assure that it lives at least as long as ExecutableNetwork
    ov::CompiledModel m_compiled_model;
};

class Lpr {
public:
    Lpr() = default;
    Lpr(ov::Core& core, const std::string& deviceName, const std::string& xmlPath, const bool autoResize,
        const std::map<std::string, std::string>& pluginConfig) :
        m_core{core} {
        slog::info << "Reading model: " << xmlPath << slog::endl;
        std::shared_ptr<ov::Model> model = m_core.read_model(xmlPath);
        logBasicModelInfo(model);

        // LPR network should have 2 inputs (and second is just a stub) and one output

        // Check inputs
        ov::OutputVector inputs = model->inputs();
        if (inputs.size() != 1 && inputs.size() != 2) {
            throw std::logic_error("LPR should have 1 or 2 inputs");
        }

        for (auto input : inputs) {
            if (input.get_shape().size() == 4)
                LprInputName = input.get_any_name();
            // LPR model that converted from Caffe have second a stub input
            if (input.get_shape().size() == 2)
                LprInputSeqName = input.get_any_name();
        }

        // Check outputs

        maxSequenceSizePerPlate = 1;

        ov::OutputVector outputs = model->outputs();
        if (outputs.size() != 1) {
            throw std::logic_error("LPR should have 1 output");
        }

        LprOutputName = outputs[0].get_any_name();

        for (size_t dim : outputs[0].get_shape()) {
            if (dim == 1) {
                continue;
            }
            if (maxSequenceSizePerPlate == 1) {
                maxSequenceSizePerPlate = dim;
            } else {
                throw std::logic_error("Every dimension of LPR output except for one must be of size 1");
            }
        }

        PrePostProcessor ppp(model);

        InputInfo& inputInfo = ppp.input(LprInputName);

        InputTensorInfo& inputTensorInfo = inputInfo.tensor();
        inputTensorInfo.set_element_type(ov::element::u8);
        if (FLAGS_auto_resize) {
            inputTensorInfo.set_layout({ "NHWC" });
        } else {
            inputTensorInfo.set_layout({ "NCHW" });
        }

        PreProcessSteps& preProcessSteps = inputInfo.preprocess();
        preProcessSteps.convert_element_type(ov::element::f32);
        if (FLAGS_auto_resize) {
            preProcessSteps.resize(ResizeAlgorithm::RESIZE_LINEAR);
        }

        InputModelInfo& inputModelInfo = inputInfo.model();
        inputModelInfo.set_layout({ "NCHW" });

        model = ppp.build();

        m_compiled_model = m_core.compile_model(model, deviceName, pluginConfig);
        logCompiledModelInfo(m_compiled_model, FLAGS_m_lpr, deviceName, "License Plate Recognition");
    }

    ov::InferRequest createInferRequest() {
        return m_compiled_model.create_infer_request();
    }

    void setImage(ov::InferRequest& inferRequest, const cv::Mat& img, const cv::Rect plateRect) {
        ov::Tensor inputTensor = inferRequest.get_tensor(LprInputName);
        ov::Shape shape = inputTensor.get_shape();
        if ((shape.size() == 4) && (3 == shape[ov::layout::channels_idx(ov::Layout({ "NHWC" }))])) {
            // autoResize is set
            ov::Tensor frameTensor = wrapMat2Tensor(img);
            ov::Coordinate p00({ 0, (size_t)plateRect.y, (size_t)plateRect.x, 0 });
            ov::Coordinate p01({ 1, (size_t)(plateRect.y + plateRect.height), (size_t)(plateRect.x + plateRect.width), 3 });
            ov::Tensor roiTensor(frameTensor, p00, p01);
            inferRequest.set_tensor(LprInputName, roiTensor);
        } else {
            const cv::Mat& vehicleImage = img(plateRect);
            matToTensor(vehicleImage, inputTensor);
        }

        if (LprInputSeqName != "") {
            ov::Tensor inputSeqTensor = inferRequest.get_tensor(LprInputSeqName);
            float* data = inputSeqTensor.data<float>();
            std::fill(data, data + inputSeqTensor.get_shape()[0], 1.0f);
        }
    }

    std::string getResults(ov::InferRequest& inferRequest) {
        static const char* const items[] = {
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                "<Anhui>", "<Beijing>", "<Chongqing>", "<Fujian>",
                "<Gansu>", "<Guangdong>", "<Guangxi>", "<Guizhou>",
                "<Hainan>", "<Hebei>", "<Heilongjiang>", "<Henan>",
                "<HongKong>", "<Hubei>", "<Hunan>", "<InnerMongolia>",
                "<Jiangsu>", "<Jiangxi>", "<Jilin>", "<Liaoning>",
                "<Macau>", "<Ningxia>", "<Qinghai>", "<Shaanxi>",
                "<Shandong>", "<Shanghai>", "<Shanxi>", "<Sichuan>",
                "<Tianjin>", "<Tibet>", "<Xinjiang>", "<Yunnan>",
                "<Zhejiang>", "<police>",
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                "U", "V", "W", "X", "Y", "Z"
        };
        std::string result;
        result.reserve(14u + 6u);  // the longest province name + 6 plate signs

        ov::Tensor lprOutputTensor = inferRequest.get_tensor(LprOutputName);
        ov::element::Type precision = lprOutputTensor.get_element_type();

        // up to 88 items per license plate, ended with "-1"
        switch (precision) {
            case ov::element::i32:
            {
                const auto data = lprOutputTensor.data<int32_t>();
                for (int i = 0; i < maxSequenceSizePerPlate; i++) {
                    int32_t val = data[i];
                    if (val == -1) {
                        break;
                    }
                    result += items[val];
                }
            }
            break;

            case ov::element::f32:
            {
                const auto data = lprOutputTensor.data<float>();
                for (int i = 0; i < maxSequenceSizePerPlate; i++) {
                    int32_t val = int32_t(data[i]);
                    if (val == -1) {
                        break;
                    }
                    result += items[val];
                }
            }
            break;

            default:
                throw std::logic_error("Not expected output blob precision");
                break;
        }
        return result;
    }

private:
    int maxSequenceSizePerPlate = 0;
    std::string LprInputName;
    std::string LprInputSeqName;
    std::string LprOutputName;
    ov::Core m_core;  // The only reason to store a device as to assure that it lives at least as long as ExecutableNetwork
    ov::CompiledModel m_compiled_model;
};
