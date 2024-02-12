// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <string>
#include <utility>
#include <vector>
#include <map>

#include "openvino/openvino.hpp"

#include "utils/common.hpp"
#include "utils/ocv_common.hpp"

class PersonDetector {
public:
    struct Result {
        std::size_t label;
        float confidence;
        cv::Rect location;
    };

    static constexpr int maxProposalCount = 200;
    static constexpr int objectSize = 7;  // Output should have 7 as a last dimension"

    PersonDetector() = default;
    PersonDetector(ov::Core& core, const std::string& deviceName, const std::string& xmlPath, const std::vector<float>& detectionTresholds,
            const bool autoResize) :
        autoResize{autoResize}, detectionTresholds{detectionTresholds}, core_{core} {
        slog::info << "Reading Person Detection model " << xmlPath << slog::endl;
        auto model = core.read_model(xmlPath);
        logBasicModelInfo(model);
        ov::OutputVector inputInfo = model->inputs();
        if (inputInfo.size() != 1) {
            throw std::logic_error("Person Detection model should have only one input");
        }

        ov::preprocess::PrePostProcessor ppp(model);
        if (autoResize) {
            ppp.input().tensor().
                set_element_type(ov::element::u8).
                set_spatial_dynamic_shape().
                set_layout({ "NHWC" });

            ppp.input().preprocess().
                convert_element_type(ov::element::f32).
                convert_layout("NCHW").
                resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);

            ppp.input().model().set_layout("NCHW");
        } else {
            ppp.input().tensor().
                set_element_type(ov::element::u8).
                set_layout({ "NCHW" });
        }

        // ---------------------------Check outputs ------------------------------------------------------
        ov::OutputVector outputInfo = model->outputs();
        if (outputInfo.size() != 1) {
            throw std::logic_error("Person Detection model should have only one output");
        }
        const ov::Shape outputShape = model->output().get_shape();
        if (maxProposalCount != outputShape[2]) {
            throw std::logic_error("unexpected ProposalCount");
        }
        if (objectSize != outputShape[3]) {
            throw std::logic_error("Output should have 7 as a last dimension");
        }
        if (outputShape.size() != 4) {
            throw std::logic_error("Incorrect output dimensions for SSD");
        }

        ppp.output().tensor().set_element_type(ov::element::f32);
        model = ppp.build();
        compiledModel = core_.compile_model(model, deviceName);
        logCompiledModelInfo(compiledModel, xmlPath, deviceName, "Person Detection");
    }

    ov::InferRequest createInferRequest() {
        return compiledModel.create_infer_request();
    }

    void setImage(ov::InferRequest& inferRequest, const cv::Mat& img) {
        ov::Tensor input = inferRequest.get_input_tensor();
        if (autoResize) {
            if (!img.isSubmatrix()) {
                // just wrap Mat object with Blob::Ptr without additional memory allocation
                ov::Tensor frameTensor = wrapMat2Tensor(img);
                inferRequest.set_input_tensor(frameTensor);
            } else {
                throw std::logic_error("Sparse matrix are not supported");
            }
        } else {
            matToTensor(img, input);
        }
    }

    std::list<Result> getResults(ov::InferRequest& inferRequest, cv::Size upscale, std::vector<std::string>& rawResults) {
        // there is no big difference if InferReq of detector from another device is passed because the processing is the same for the same topology
        std::list<Result> results;
        const float* const detections = inferRequest.get_output_tensor().data<float>();

        // pretty much regular SSD post-processing
        for (int i = 0; i < maxProposalCount; i++) {
            float image_id = detections[i * objectSize + 0];  // in case of batch
            if (image_id < 0) {  // indicates end of detections
                break;
            }
            auto label = static_cast<decltype(detectionTresholds.size())>(detections[i * objectSize + 1]);
            float confidence = detections[i * objectSize + 2];
            if (0  < detectionTresholds.size() && confidence < detectionTresholds[0]) {
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
    bool autoResize;
    std::vector<float> detectionTresholds;
    ov::Core core_;  // The only reason to store a plugin as to assure that it lives at least as long as CompiledModel
    ov::CompiledModel compiledModel;
};

class ReId {
public:
    ReId() = default;
    ReId(ov::Core& core, const std::string& deviceName, const std::string& xmlPath, const bool autoResize) :
        autoResize {autoResize},
        core_{core} {
        slog::info << "Reading Person Re-ID model " << xmlPath << slog::endl;
        auto model = core.read_model(xmlPath);
        logBasicModelInfo(model);
        /** Re-ID model should have only one input and one output **/
        // ---------------------------Check inputs ------------------------------------------------------
        ov::OutputVector inputInfo = model->inputs();
        if (inputInfo.size() != 1) {
            throw std::logic_error("Re-ID model should have only one input");
        }

        ov::preprocess::PrePostProcessor ppp(model);
        if (autoResize) {
            ppp.input().tensor().
                set_element_type(ov::element::u8).
                set_spatial_dynamic_shape().
                set_layout({ "NHWC" });

            ppp.input().preprocess().
                convert_element_type(ov::element::f32).
                convert_layout("NCHW").
                resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);

            ppp.input().model().set_layout("NCHW");
        }
        else {
            ppp.input().tensor().
                set_element_type(ov::element::u8).
                set_layout({ "NCHW" });
        }
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        ov::OutputVector outputInfo = model->outputs();
        if (outputInfo.size() != 1) {
            throw std::logic_error("Re-ID model should have 1 output");
        }
        const ov::Shape outputShape = model->output().get_shape();
        if (outputShape.size() != 2) {
            throw std::logic_error("Incorrect output dimensions for Re-ID");
        }

        reidLen = (int)outputShape[1];
        ppp.output().tensor().set_element_type(ov::element::f32);
        model = ppp.build();
        compiledModel = core_.compile_model(model, deviceName);
        logCompiledModelInfo(compiledModel, xmlPath, deviceName, "Person Re-ID");
    }

    ov::InferRequest createInferRequest() {
        return compiledModel.create_infer_request();
    }

    void setImage(ov::InferRequest& inferRequest, const cv::Mat& img, const cv::Rect personRect) {
        ov::Tensor input = inferRequest.get_input_tensor();
        if (autoResize) {
            ov::Tensor frameTensor = wrapMat2Tensor(img);
            ov::Shape tensorShape = frameTensor.get_shape();
            ov::Layout layout("NHWC");
            const size_t batch = tensorShape[ov::layout::batch_idx(layout)];
            const size_t channels = tensorShape[ov::layout::channels_idx(layout)];
            ov::Tensor roiTensor(frameTensor, {0, static_cast<size_t>(personRect.y),  static_cast<size_t>(personRect.x), 0},
                {batch, static_cast<size_t>(personRect.y) + static_cast<size_t>(personRect.height),
                static_cast<size_t>(personRect.x) + static_cast<size_t>(personRect.width), channels});
            inferRequest.set_input_tensor(roiTensor);
        } else {
            const cv::Mat& personImage = img(personRect);
            matToTensor(personImage, input);
        }
    }

    std::vector<float> getResults(ov::InferRequest& inferRequest) {
        std::vector<float> result;
        const float* const reids = inferRequest.get_output_tensor().data<float>();
        for (int i = 0; i < reidLen; i++) {
            result.push_back(reids[i]);
        }
        return result;
    }

private:
    bool autoResize;
    int reidLen;
    ov::Core core_;  // The only reason to store a device as to assure that it lives at least as long as  CompiledModel
    ov::CompiledModel compiledModel;
};
