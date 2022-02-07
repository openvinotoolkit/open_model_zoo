// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <openvino/openvino.hpp>

#include <utils/slog.hpp>
#include <utils/common.hpp>

#include "cnn.hpp"

CnnBase::CnnBase(const Config& config,
    const ov::Core& core,
    const std::string& device_name) :
    config(config), core(core), device_name(device_name) {}

void CnnBase::Load() {
    slog::info << "Reading model: " << config.path_to_model << slog::endl;
    auto model = core.read_model(config.path_to_model);
    logBasicModelInfo(model);

    if (model->inputs().size() != 1) {
        throw std::logic_error("Demo supports topologies only with 1 input");
    }

    ov::preprocess::PrePostProcessor ppp(model);
    input_layout = { "NCHW" };
    ppp.input().tensor().
        set_element_type(ov::element::u8).
        set_layout({ "NCHW" });

    ppp.input().model().set_layout(input_layout);

    if (model->outputs().size() != 1) {
        throw std::runtime_error("Demo supports topologies only with 1 output");
    }
    ppp.output().tensor().set_element_type(ov::element::f32);

    model = ppp.build();

    input_shape = model->input().get_shape();
    input_shape[ov::layout::batch_idx(input_layout)] = config.max_batch_size;
    output_shape = model->output().get_shape();
    ov::set_batch(model, { 1, int64(config.max_batch_size) });

    compiled_model = core.compile_model(model, device_name);
    logCompiledModelInfo(compiled_model, config.path_to_model, device_name, modelType);

    infer_request = compiled_model.create_infer_request();
    input_tensor = infer_request.get_input_tensor();
    output_tensor = infer_request.get_output_tensor();
}

void CnnBase::InferBatch(
    const std::vector<cv::Mat>& frames,
    const std::function<void(const ov::Tensor&, size_t)>&  fetch_results) const {
    size_t num_imgs = frames.size();
    input_tensor.set_shape(input_shape);
    for (size_t i = 0; i < num_imgs; ++i) {
        matToTensor(frames[i], input_tensor, i);
    }
    infer_request.set_input_tensor(ov::Tensor(input_tensor, {0, 0, 0, 0}, {num_imgs, input_shape[ov::layout::channels_idx(input_layout)],
        input_shape[ov::layout::height_idx(input_layout)], input_shape[ov::layout::width_idx(input_layout)]}));
    infer_request.infer();
    fetch_results(infer_request.get_output_tensor(), num_imgs);
}

VectorCNN::VectorCNN(const Config& config,
    const ov::Core& core,
    const std::string& deviceName)
    : CnnBase(config, core, deviceName) {
    Load();
    result_size = std::accumulate(std::next(output_shape.begin(), 1), output_shape.end(), 1, std::multiplies<int>());
}

void VectorCNN::Compute(const cv::Mat& frame,
    cv::Mat* vector, cv::Size out_shape) const {
    std::vector<cv::Mat> output;
    Compute({ frame }, &output, out_shape);
    *vector = output[0];
}

void VectorCNN::Compute(const std::vector<cv::Mat>& images, std::vector<cv::Mat>* vectors,
    cv::Size out_shape) const {
    if (images.empty()) {
        return;
    }
    vectors->clear();
    auto results_fetcher = [vectors, out_shape](const ov::Tensor& tensor, size_t batch_size) {
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
            if (out_shape != cv::Size()) {
                tensor_wrapper = tensor_wrapper.reshape(1, { out_shape.height, out_shape.width });
            }
            tensor_wrapper.copyTo(vectors->back());
        }

    };
    InferBatch(images, results_fetcher);
}
