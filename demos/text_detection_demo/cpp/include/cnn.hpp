// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "openvino/openvino.hpp"


class Cnn {
public:
    Cnn(const std::string& modelPath, const std::string& modelType, const std::string& deviceName,
        ov::Core& core, const cv::Size& new_input_resolution = cv::Size(), bool use_auto_resize = false);
    virtual ~Cnn() = default;

    virtual std::map<std::string, ov::Tensor> Infer(const cv::Mat& frame) = 0;

    size_t ncalls() const { return m_ncalls; }
    double time_elapsed() const { return m_time_elapsed; }
    const cv::Size& input_size() const { return m_input_size; }

protected:
    const std::string m_modelPath;
    const std::string m_modelType;
    const std::string m_deviceName;
    ov::Core& m_core;
    cv::Size m_new_input_resolution;
    bool use_auto_resize;
    size_t m_channels;
    cv::Size m_input_size;
    std::string m_input_name;
    std::vector<std::string> m_output_names;
    ov::Layout m_modelLayout;
    ov::InferRequest m_infer_request;
    std::shared_ptr<ov::Model> m_model;

    double m_time_elapsed;
    size_t m_ncalls;
};
