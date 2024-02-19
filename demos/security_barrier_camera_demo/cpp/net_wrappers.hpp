// Copyright (C) 2018-2024 Intel Corporation
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
            const bool autoResize) :
        m_autoResize(autoResize), m_detectionTresholds{detectionTresholds} {
        slog::info << "Reading model: " << xmlPath << slog::endl;
        std::shared_ptr<ov::Model> model = core.read_model(xmlPath);
        logBasicModelInfo(model);

        // Check model inputs and outputs

        ov::OutputVector inputs = model->inputs();
        if (inputs.size() != 1) {
            throw std::logic_error("Detector should have only one input");
        }

        m_detectorInputName = model->input().get_any_name();

        ov::Layout modelLayout = ov::layout::get_layout(model->input());
        if (modelLayout.empty())
            modelLayout = {"NCHW"};

        ov::OutputVector outputs = model->outputs();
        if (outputs.size() != 1) {
            throw std::logic_error("Vehicle Detection network should have only one output");
        }

        ov::Output<ov::Node> output = outputs[0];

        m_detectorOutputName = output.get_any_name();
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

        ov::preprocess::PrePostProcessor ppp(model);

        ov::preprocess::InputInfo& inputInfo = ppp.input();

        ov::preprocess::InputTensorInfo& inputTensorInfo = inputInfo.tensor();
        // configure desired input type and layout, the
        // use preprocessor to convert to actual model input type and layout
        inputTensorInfo.set_element_type(ov::element::u8);
        inputTensorInfo.set_layout({"NHWC"});
        if (autoResize) {
            inputTensorInfo.set_spatial_dynamic_shape();
        }

        ov::preprocess::InputModelInfo& inputModelInfo = inputInfo.model();
        inputModelInfo.set_layout(modelLayout);

        ov::preprocess::PreProcessSteps& preProcessSteps = inputInfo.preprocess();
        preProcessSteps.convert_layout(modelLayout);
        preProcessSteps.convert_element_type(ov::element::f32);
        if (autoResize) {
            preProcessSteps.resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
        }

        model = ppp.build();

        slog::info << "Preprocessor configuration: " << slog::endl;
        slog::info << ppp << slog::endl;

        m_compiled_model = core.compile_model(model, deviceName);
        logCompiledModelInfo(m_compiled_model, xmlPath, deviceName, "Vehicle And License Plate Detection");
    }

    ov::InferRequest createInferRequest() {
        return m_compiled_model.create_infer_request();
    }

    void setImage(ov::InferRequest& inferRequest, const cv::Mat& img) {
        ov::Tensor inputTensor = inferRequest.get_tensor(m_detectorInputName);
        ov::Shape shape = inputTensor.get_shape();
        if (m_autoResize) {
            if (!img.isSubmatrix()) {
                // just wrap Mat object with Tensor without additional memory allocation
                ov::Tensor frameTensor = wrapMat2Tensor(img);
                inferRequest.set_tensor(m_detectorInputName, frameTensor);
            } else {
                throw std::logic_error("Sparse matrix are not supported");
            }
        } else {
            // resize and copy data from image to tensor using OpenCV
            resize2tensor(img, inputTensor);
        }
    }

    std::list<Result> getResults(ov::InferRequest& inferRequest, cv::Size upscale, std::vector<std::string>& rawResults) {
        // there is no big difference if InferReq of detector from another device is passed
        // because the processing is the same for the same topology
        std::list<Result> results;
        ov::Tensor output_tensor = inferRequest.get_tensor(m_detectorOutputName);
        const float* const detections = output_tensor.data<float>();
        // pretty much regular SSD post-processing
        for (int i = 0; i < maxProposalCount; i++) {
            float image_id = detections[i * objectSize + 0]; // in case of batch
            if (image_id < 0) { // indicates end of detections
                break;
            }
            size_t label = static_cast<decltype(m_detectionTresholds.size())>(detections[i * objectSize + 1]);
            float confidence = detections[i * objectSize + 2];
            if (label - 1 < m_detectionTresholds.size() && confidence < m_detectionTresholds[label - 1]) {
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
    bool m_autoResize;
    std::vector<float> m_detectionTresholds;
    std::string m_detectorInputName;
    std::string m_detectorOutputName;
    ov::CompiledModel m_compiled_model;
};

class VehicleAttributesClassifier {
public:
    VehicleAttributesClassifier() = default;
    VehicleAttributesClassifier(ov::Core& core, const std::string& deviceName,
        const std::string& xmlPath, const bool autoResize) :
        m_autoResize(autoResize) {
        slog::info << "Reading model: " << xmlPath << slog::endl;
        std::shared_ptr<ov::Model> model = core.read_model(xmlPath);
        logBasicModelInfo(model);

        ov::OutputVector inputs = model->inputs();
        if (inputs.size() != 1) {
            throw std::logic_error("Vehicle Attribs topology should have only one input");
        }

        m_attributesInputName = model->input().get_any_name();

        ov::Layout modelLayout = ov::layout::get_layout(model->input());
        if (modelLayout.empty())
            modelLayout = {"NCHW"};

        ov::OutputVector outputs = model->outputs();
        if (outputs.size() != 2) {
            throw std::logic_error("Vehicle Attribs Network expects networks having two outputs");
        }

        // color is the first output
        m_outputNameForColor = outputs[0].get_any_name();
        // type is the second output.
        m_outputNameForType = outputs[1].get_any_name();

        ov::preprocess::PrePostProcessor ppp(model);

        ov::preprocess::InputInfo& inputInfo = ppp.input();

        ov::preprocess::InputTensorInfo& inputTensorInfo = inputInfo.tensor();
        // configure desired input type and layout, the
        // use preprocessor to convert to actual model input type and layout
        inputTensorInfo.set_element_type(ov::element::u8);
        inputTensorInfo.set_layout({"NHWC"});
        if (autoResize) {
            inputTensorInfo.set_spatial_dynamic_shape();
        }

        ov::preprocess::PreProcessSteps& preProcessSteps = inputInfo.preprocess();
        preProcessSteps.convert_layout(modelLayout);
        preProcessSteps.convert_element_type(ov::element::f32);
        if (autoResize) {
            preProcessSteps.resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
        }

        ov::preprocess::InputModelInfo& inputModelInfo = inputInfo.model();
        inputModelInfo.set_layout(modelLayout);

        model = ppp.build();

        slog::info << "Preprocessor configuration: " << slog::endl;
        slog::info << ppp << slog::endl;

        m_compiled_model = core.compile_model(model, deviceName);
        logCompiledModelInfo(m_compiled_model, xmlPath, deviceName, "Vehicle Attributes Recognition");
    }

    ov::InferRequest createInferRequest() {
        return m_compiled_model.create_infer_request();
    }

    void setImage(ov::InferRequest& inferRequest, const cv::Mat& img, const cv::Rect vehicleRect) {
        ov::Tensor inputTensor = inferRequest.get_tensor(m_attributesInputName);
        ov::Shape shape = inputTensor.get_shape();
        if (m_autoResize) {
            ov::Tensor frameTensor = wrapMat2Tensor(img);
            ov::Coordinate p00({ 0, (size_t)vehicleRect.y, (size_t)vehicleRect.x, 0 });
            ov::Coordinate p01({ 1, (size_t)(vehicleRect.y + vehicleRect.height), (size_t)vehicleRect.x + vehicleRect.width, 3 });
            ov::Tensor roiTensor(frameTensor, p00, p01);

            inferRequest.set_tensor(m_attributesInputName, roiTensor);
        } else {
            const cv::Mat& vehicleImage = img(vehicleRect);
            resize2tensor(vehicleImage, inputTensor);
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
        ov::Tensor colorsTensor = inferRequest.get_tensor(m_outputNameForColor);
        const float* colorsValues = colorsTensor.data<float>();

        // 4 possible types for each vehicle and we should select the one with the maximum probability
        ov::Tensor typesTensor = inferRequest.get_tensor(m_outputNameForType);
        const float* typesValues = typesTensor.data<float>();

        const auto color_id = std::max_element(colorsValues, colorsValues + 7) - colorsValues;
        const auto  type_id = std::max_element(typesValues,  typesValues  + 4) - typesValues;

        return std::pair<std::string, std::string>(colors[color_id], types[type_id]);
    }

private:
    bool m_autoResize;
    std::string m_attributesInputName;
    std::string m_outputNameForColor;
    std::string m_outputNameForType;
    ov::CompiledModel m_compiled_model;
};

class Lpr {
public:
    Lpr() = default;
    Lpr(ov::Core& core, const std::string& deviceName, const std::string& xmlPath, const bool autoResize) :
        m_autoResize(autoResize) {
        slog::info << "Reading model: " << xmlPath << slog::endl;
        std::shared_ptr<ov::Model> model = core.read_model(xmlPath);
        logBasicModelInfo(model);

        // LPR network should have 2 inputs (and second is just a stub) and one output

        // Check inputs
        ov::OutputVector inputs = model->inputs();
        if (inputs.size() != 1 && inputs.size() != 2) {
            throw std::logic_error("LPR should have 1 or 2 inputs");
        }

        for (auto input : inputs) {
            if (input.get_shape().size() == 4) {
                m_LprInputName = input.get_any_name();
                m_modelLayout = ov::layout::get_layout(input);
                if (m_modelLayout.empty())
                    m_modelLayout = {"NCHW"};
            }
            // LPR model that converted from Caffe have second a stub input
            if (input.get_shape().size() == 2)
                m_LprInputSeqName = input.get_any_name();
        }

        // Check outputs

        m_maxSequenceSizePerPlate = 1;

        ov::OutputVector outputs = model->outputs();
        if (outputs.size() != 1) {
            throw std::logic_error("LPR should have 1 output");
        }

        m_LprOutputName = outputs[0].get_any_name();

        for (size_t dim : outputs[0].get_shape()) {
            if (dim == 1) {
                continue;
            }
            if (m_maxSequenceSizePerPlate == 1) {
                m_maxSequenceSizePerPlate = dim;
            } else {
                throw std::logic_error("Every dimension of LPR output except for one must be of size 1");
            }
        }

        ov::preprocess::PrePostProcessor ppp(model);

        ov::preprocess::InputInfo& inputInfo = ppp.input(m_LprInputName);

        ov::preprocess::InputTensorInfo& inputTensorInfo = inputInfo.tensor();
        // configure desired input type and layout, the
        // use preprocessor to convert to actual model input type and layout
        inputTensorInfo.set_element_type(ov::element::u8);
        inputTensorInfo.set_layout({"NHWC"});
        if (autoResize) {
            inputTensorInfo.set_spatial_dynamic_shape();
        }

        ov::preprocess::PreProcessSteps& preProcessSteps = inputInfo.preprocess();
        preProcessSteps.convert_layout(m_modelLayout);
        preProcessSteps.convert_element_type(ov::element::f32);
        if (autoResize) {
            preProcessSteps.resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
        }

        ov::preprocess::InputModelInfo& inputModelInfo = inputInfo.model();
        inputModelInfo.set_layout(m_modelLayout);

        model = ppp.build();

        slog::info << "Preprocessor configuration: " << slog::endl;
        slog::info << ppp << slog::endl;

        m_compiled_model = core.compile_model(model, deviceName);
        logCompiledModelInfo(m_compiled_model, xmlPath, deviceName, "License Plate Recognition");
    }

    ov::InferRequest createInferRequest() {
        return m_compiled_model.create_infer_request();
    }

    void setImage(ov::InferRequest& inferRequest, const cv::Mat& img, const cv::Rect plateRect) {
        ov::Tensor inputTensor = inferRequest.get_tensor(m_LprInputName);
        ov::Shape shape = inputTensor.get_shape();
        if ((shape.size() == 4) && m_autoResize) {
            // autoResize is set
            ov::Tensor frameTensor = wrapMat2Tensor(img);
            ov::Coordinate p00({ 0, (size_t)plateRect.y, (size_t)plateRect.x, 0 });
            ov::Coordinate p01({ 1, (size_t)(plateRect.y + plateRect.height), (size_t)(plateRect.x + plateRect.width), 3 });
            ov::Tensor roiTensor(frameTensor, p00, p01);
            inferRequest.set_tensor(m_LprInputName, roiTensor);
        } else {
            const cv::Mat& vehicleImage = img(plateRect);
            resize2tensor(vehicleImage, inputTensor);
        }

        if (m_LprInputSeqName != "") {
            ov::Tensor inputSeqTensor = inferRequest.get_tensor(m_LprInputSeqName);
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

        ov::Tensor lprOutputTensor = inferRequest.get_tensor(m_LprOutputName);
        ov::element::Type precision = lprOutputTensor.get_element_type();

        // up to 88 items per license plate, ended with "-1"
        switch (precision) {
            case ov::element::i32:
            {
                const auto data = lprOutputTensor.data<int32_t>();
                for (int i = 0; i < m_maxSequenceSizePerPlate; i++) {
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
                for (int i = 0; i < m_maxSequenceSizePerPlate; i++) {
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
    bool m_autoResize;
    int m_maxSequenceSizePerPlate = 0;
    std::string m_LprInputName;
    std::string m_LprInputSeqName;
    std::string m_LprOutputName;
    ov::Layout m_modelLayout;
    ov::CompiledModel m_compiled_model;
};
