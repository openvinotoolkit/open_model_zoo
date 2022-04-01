// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cnn.hpp"

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

#include <utils/common.hpp>
#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>

BaseModel::BaseModel(const Config& config, const ov::Core& core, const std::string& device_name)
    : config(config),
      core(core),
      device_name(device_name) {}

void BaseModel::Load() {
    slog::info << "Reading model: " << config.path_to_model << slog::endl;
    auto model = core.read_model(config.path_to_model);
    logBasicModelInfo(model);

    if (model->inputs().size() != 1) {
        throw std::logic_error("Demo supports topologies with only 1 input");
    }

    ov::preprocess::PrePostProcessor ppp(model);
    input_layout = getLayoutFromShape(model->input().get_shape());
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout({"NCHW"});

    ppp.input().model().set_layout(input_layout);

    if (model->outputs().size() != 1) {
        throw std::runtime_error("Demo supports topologies with only 1 output");
    }
    ppp.output().tensor().set_element_type(ov::element::f32);

    model = ppp.build();

    input_shape = model->input().get_shape();
    input_shape[ov::layout::batch_idx(input_layout)] = config.max_batch_size;
    output_shape = model->output().get_shape();
    ov::set_batch(model, {1, static_cast<int64>(config.max_batch_size)});

    compiled_model = core.compile_model(model, device_name);
    logCompiledModelInfo(compiled_model, config.path_to_model, device_name, modelType);

    infer_request = compiled_model.create_infer_request();
    input_tensor = infer_request.get_input_tensor();
    output_tensor = infer_request.get_output_tensor();
}

void BaseModel::InferBatch(const std::vector<cv::Mat>& frames,
                           const std::function<void(const ov::Tensor&, size_t)>& fetch_results) const {
    size_t num_imgs = frames.size();
    input_tensor.set_shape(input_shape);
    for (size_t batch_i = 0; batch_i < num_imgs;) {
        size_t batch_size = std::min(num_imgs - batch_i, (size_t)config.max_batch_size);
        for (size_t b = 0; b < batch_size; ++b) {
            matToTensor(frames[batch_i + b], input_tensor, b);
        }
        infer_request.set_input_tensor(ov::Tensor(input_tensor,
                                                  {0, 0, 0, 0},
                                                  {batch_size,
                                                   input_shape[ov::layout::channels_idx(input_layout)],
                                                   input_shape[ov::layout::height_idx(input_layout)],
                                                   input_shape[ov::layout::width_idx(input_layout)]}));
        infer_request.infer();
        fetch_results(infer_request.get_output_tensor(), batch_size);
        batch_i += batch_size;
    }
}

VectorCNN::VectorCNN(const Config& config, const ov::Core& core, const std::string& deviceName)
    : BaseModel(config, core, deviceName) {
    Load();
    result_size = std::accumulate(std::next(output_shape.begin(), 1), output_shape.end(), 1, std::multiplies<int>());
}

void VectorCNN::Compute(const cv::Mat& frame, cv::Mat* vector, cv::Size out_shape) const {
    std::vector<cv::Mat> output;
    Compute({frame}, &output, out_shape);
    *vector = output[0];
}

void VectorCNN::Compute(const std::vector<cv::Mat>& images, std::vector<cv::Mat>* vectors, cv::Size out_shape) const {
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
            cv::Mat tensor_wrapper(out_tensor.size[1],
                                   1,
                                   CV_32F,
                                   reinterpret_cast<void*>((out_tensor.ptr<float>(0) + b * out_tensor.size[1])));
            vectors->emplace_back();
            if (out_shape != cv::Size()) {
                tensor_wrapper = tensor_wrapper.reshape(1, {out_shape.height, out_shape.width});
            }
            tensor_wrapper.copyTo(vectors->back());
        }
    };
    InferBatch(images, results_fetcher);
}
