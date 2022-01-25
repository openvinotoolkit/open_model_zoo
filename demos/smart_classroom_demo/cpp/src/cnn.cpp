// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cnn.hpp"

#include <string>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <inference_engine.hpp>

#include <map>

#include "openvino/core/layout.hpp"
#include "openvino/openvino.hpp"

CnnDLSDKBase::CnnDLSDKBase(const Config& config) : config_(config) {}

void CnnDLSDKBase::Load() {
    auto cnnNetwork = config_.ie.read_model(config_.path_to_model);

    const int currentBatchSize = cnnNetwork->input().get_shape()[0];

    ov::OutputVector in = cnnNetwork->inputs();
    if (in.size() != 1) {
        throw std::runtime_error("Network should have only one input");
    }

    ov::preprocess::PrePostProcessor ppp(cnnNetwork);
    ppp.input().tensor().
      set_element_type(ov::element::f32).
      set_layout({"NCHW"});
    input_blob_name_ = cnnNetwork->input().get_any_name();
    ov::OutputVector outputs = cnnNetwork->outputs();
    for (auto&& item : outputs) {
      ppp.output(*item.get_names().begin()).tensor().set_element_type(ov::element::f32);
      output_blobs_names_.push_back(item.get_any_name());
    }
    cnnNetwork = ppp.build();
    ov::set_batch(cnnNetwork, config_.max_batch_size);

    try {
        compiled_model_ = config_.ie.compile_model(cnnNetwork, config_.deviceName);
    } catch (const ov::Exception&) {
        ov::set_batch(cnnNetwork, 1);
        compiled_model_ = config_.ie.compile_model(cnnNetwork, config_.deviceName);
    }
    logExecNetworkInfo(compiled_model_, config_.path_to_model, config_.deviceName, config_.model_type);
    infer_request_ = compiled_model_.create_infer_request();
}

void CnnDLSDKBase::InferBatch(
        const std::vector<cv::Mat>& frames,
        const std::function<void(const std::map<std::string, ov::Tensor>&, size_t)>& fetch_results) const {
    ov::Tensor input = infer_request_.get_tensor(input_blob_name_);
    const size_t batch_size = input.get_shape()[0];

    size_t num_imgs = frames.size();
    for (size_t batch_i = 0; batch_i < num_imgs; batch_i += batch_size) {
        const size_t current_batch_size = std::min(batch_size, num_imgs - batch_i);
        for (size_t b = 0; b < current_batch_size; b++) {
            matToTensor(frames[batch_i + b], input, b);
        }

        infer_request_.infer();
        std::map<std::string, ov::Tensor> tensors;

        for (const auto& name : output_blobs_names_)  {
            tensors[name] = infer_request_.get_tensor(name);
        }
        fetch_results(tensors, current_batch_size);
    }
}

void CnnDLSDKBase::Infer(const cv::Mat& frame,
                         const std::function<void(const std::map<std::string, ov::Tensor>&, size_t)>& fetch_results) const {
    InferBatch({frame}, fetch_results);
}

VectorCNN::VectorCNN(const Config& config)
        : CnnDLSDKBase(config) {
    Load();
    if (output_blobs_names_.size() != 1) {
        throw std::runtime_error("Demo supports topologies only with 1 output");
    }
}

void VectorCNN::Compute(const cv::Mat& frame,
                                     cv::Mat* vector, cv::Size outp_shape) const {
    std::vector<cv::Mat> output;
    Compute({frame}, &output, outp_shape);
    *vector = output[0];
}

void VectorCNN::Compute(const std::vector<cv::Mat>& images, std::vector<cv::Mat>* vectors,
                                     cv::Size outp_shape) const {
    if (images.empty()) {
        return;
    }
    vectors->clear();
    auto results_fetcher = [vectors, outp_shape](const std::map<std::string, ov::Tensor>& outputs, size_t batch_size) {
        for (auto&& item : outputs) {
            ov::Tensor tensor = item.second;
            ov::Shape ie_output_dims = tensor.get_shape();
            std::vector<int> tensor_sizes(ie_output_dims.size(), 0);
            for (size_t i = 0; i < tensor_sizes.size(); ++i) {
                tensor_sizes[i] = ie_output_dims[i];
            }
            cv::Mat out_tensor(tensor_sizes, CV_32F, tensor.data<float>());
            for (size_t b = 0; b < batch_size; b++) {
                cv::Mat tensor_wrapper(out_tensor.size[1], 1, CV_32F,
                                     reinterpret_cast<void*>((out_tensor.ptr<float>(0) + b * out_tensor.size[1])));
                vectors->emplace_back();
                if (outp_shape != cv::Size())
                    tensor_wrapper = tensor_wrapper.reshape(1, {outp_shape.height, outp_shape.width});
                tensor_wrapper.copyTo(vectors->back());
            }
        }
    };
    InferBatch(images, results_fetcher);
}
