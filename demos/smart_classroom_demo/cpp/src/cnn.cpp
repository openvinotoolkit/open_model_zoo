// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "openvino/openvino.hpp"

#include "cnn.hpp"

CnnDLSDKBase::CnnDLSDKBase(const Config& config) : m_config(config) {}

void CnnDLSDKBase::Load() {
    slog::info << "Reading model: " << m_config.m_path_to_model << slog::endl;
    std::shared_ptr<ov::Model> model = m_config.m_core.read_model(m_config.m_path_to_model);
    logBasicModelInfo(model);

    ov::OutputVector inputs = model->inputs();
    if (inputs.size() != 1) {
        throw std::runtime_error("Network should have only one input");
    }

    ov::Layout desiredLayout = {"NHWC"};

    m_input_tensor_name = model->input().get_any_name();

    m_modelLayout = ov::layout::get_layout(model->input());
    if (m_modelLayout.empty())
        m_modelLayout = {"NCHW"};

    m_modelShape = model->input().get_shape();

    ov::OutputVector outputs = model->outputs();
    for (auto& item : outputs) {
        const std::string name = item.get_any_name();
        m_output_tensors_names.push_back(name);
    }

    ov::preprocess::PrePostProcessor ppp(model);

    ppp.input().tensor()
        .set_element_type(ov::element::u8)
        .set_layout(desiredLayout);
    ppp.input().preprocess()
        .convert_layout(m_modelLayout)
        .convert_element_type(ov::element::f32);
    ppp.input().model().set_layout(m_modelLayout);

    model = ppp.build();

    slog::info << "PrePostProcessor configuration:" << slog::endl;
    slog::info << ppp << slog::endl;

    // allow 1...max_batch range, actual will be set at inference time
    ov::set_batch(model, {1, m_config.m_max_batch_size});

    m_modelShape[ov::layout::batch_idx(m_modelLayout)] = m_config.m_max_batch_size;

    m_compiled_model = m_config.m_core.compile_model(model, m_config.m_deviceName);
    logCompiledModelInfo(m_compiled_model, m_config.m_path_to_model, m_config.m_deviceName, m_config.m_model_type);

    m_infer_request = m_compiled_model.create_infer_request();
}

void CnnDLSDKBase::InferBatch(
        const std::vector<cv::Mat>& frames,
        const std::function<void(const std::map<std::string, ov::Tensor>&, size_t)>& fetch_results) const {
    ov::Tensor input_tensor = m_infer_request.get_tensor(m_input_tensor_name);

    size_t num_imgs = frames.size();

    size_t h = m_modelShape[ov::layout::height_idx(m_modelLayout)];
    size_t w = m_modelShape[ov::layout::width_idx(m_modelLayout)];
    size_t c = m_modelShape[ov::layout::channels_idx(m_modelLayout)];

    // shrink tensor from default m_config.m_max_batch_size to actual num of images;
    input_tensor.set_shape({ num_imgs, h, w, c });

    for (size_t i = 0; i < num_imgs; i++) {
        resize2tensor(frames[i], ov::Tensor{ input_tensor, {i, 0, 0, 0}, {i + 1, h, w, c} });
    }

    m_infer_request.set_input_tensor(ov::Tensor{ input_tensor, { 0, 0, 0, 0 }, {num_imgs, h, w, c} });

    m_infer_request.infer();

    std::map<std::string, ov::Tensor> output_tensors;
    for (const auto& output_tensor_name : m_output_tensors_names) {
        output_tensors[output_tensor_name] = m_infer_request.get_tensor(output_tensor_name);
    }

    fetch_results(output_tensors, num_imgs);
}

VectorCNN::VectorCNN(const Config& config) : CnnDLSDKBase(config) {
    Load();
    if (m_output_tensors_names.size() != 1) {
        throw std::runtime_error("Demo supports topologies only with 1 output");
    }
}

void VectorCNN::Compute(const cv::Mat& frame, cv::Mat* vector, cv::Size outp_shape) const {
    std::vector<cv::Mat> output;
    Compute({frame}, &output, outp_shape);
    *vector = output[0];
}

void VectorCNN::Compute(const std::vector<cv::Mat>& images, std::vector<cv::Mat>* vectors, cv::Size outp_shape) const {
    if (images.empty()) {
        return;
    }
    vectors->clear();
    auto results_fetcher = [vectors, outp_shape](const std::map<std::string, ov::Tensor>& outputs, size_t batch_size) {
        for (auto& output : outputs) {
            ov::Tensor tensor = output.second;
            ov::Shape shape = tensor.get_shape();
            std::vector<int> tensor_sizes(shape.size(), 0);
            for (size_t i = 0; i < tensor_sizes.size(); ++i) {
                tensor_sizes[i] = shape[i];
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

int VectorCNN::maxBatchSize() const {
    return m_config.m_max_batch_size;
}
