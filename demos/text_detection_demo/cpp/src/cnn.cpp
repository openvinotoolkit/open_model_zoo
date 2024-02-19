// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <map>
#include <string>

#include "utils/common.hpp"
#include "utils/ocv_common.hpp"

#include "cnn.hpp"

Cnn::Cnn(
    const std::string& modelPath, const std::string& modelType, const std::string& deviceName,
    ov::Core& core, const cv::Size& new_input_resolution, bool use_auto_resize) :
    m_modelPath(modelPath), m_modelType(modelType), m_deviceName(deviceName),
    m_core(core), m_new_input_resolution(new_input_resolution), use_auto_resize(use_auto_resize),
    m_channels(0), m_time_elapsed(0), m_ncalls(0)
{
    slog::info << "Reading model: " << m_modelPath << slog::endl;
    m_model = m_core.read_model(m_modelPath);
    logBasicModelInfo(m_model);

    ov::OutputVector inputs = m_model->inputs();
    if (inputs.size() != 1) {
        throw std::runtime_error("The model should have only one input");
    }

    ov::Shape input_shape = m_model->input().get_shape();
    if (input_shape.size() != 4) {
        throw std::runtime_error("The model should have 4-dimensional input");
    }

    ov::Layout input_layout = ov::layout::get_layout(m_model->input());
    if (input_layout.empty()) {
        // prev release model has NCHW layout but it was not specified at IR
        input_layout = { "NCHW" };
    }

    m_modelLayout = input_layout;

    input_shape[ov::layout::batch_idx(input_layout)] = 1;

    m_input_name = m_model->input().get_any_name();

    // Changing input shape if it is needed
    if (m_new_input_resolution != cv::Size()) {
        input_shape[ov::layout::height_idx(input_layout)] = static_cast<size_t>(m_new_input_resolution.height);
        input_shape[ov::layout::width_idx(input_layout)] = static_cast<size_t>(m_new_input_resolution.width);
    }

    m_model->reshape({ {m_input_name, input_shape} });

    m_channels = input_shape[ov::layout::channels_idx(input_layout)];
    m_input_size = cv::Size(int(input_shape[ov::layout::width_idx(input_layout)]), int(input_shape[ov::layout::height_idx(input_layout)]));

    // Collect output names
    ov::OutputVector outputs = m_model->outputs();
    for (const ov::Output<ov::Node>& output : outputs) {
        m_output_names.push_back(output.get_any_name());
    }

    // Configuring input and output
    ov::preprocess::PrePostProcessor ppp(m_model);

    // we'd like to pass input image (NHWC, u8) to model input
    // and let OpenVINO do necessary conversions

    ppp.input().tensor()
        .set_element_type(ov::element::u8)
        .set_layout({ "NHWC" });

    if (use_auto_resize) {
        ppp.input().tensor()
            .set_spatial_dynamic_shape();

        ppp.input().preprocess()
            .convert_element_type(ov::element::f32)
            .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
    }
    ppp.input().model().set_layout(input_layout);

    m_model = ppp.build();
    slog::info << "Preprocessor configuration: " << slog::endl;
    slog::info << ppp << slog::endl;

    // Loading model to the device
    ov::CompiledModel compiled_model = m_core.compile_model(m_model, m_deviceName);
    logCompiledModelInfo(compiled_model, m_modelPath, m_deviceName, m_modelType);

    // Creating infer request
    m_infer_request = compiled_model.create_infer_request();
}
