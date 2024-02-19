// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "openvino/openvino.hpp"

#include "cnn.hpp"

class TextDetector : public Cnn {
public:
    TextDetector(const std::string& model_path, const std::string& model_type, const std::string& deviceName,
        ov::Core& core, const cv::Size& new_input_resolution = cv::Size(), bool use_auto_resize = false) :
        Cnn(model_path, model_type, deviceName, core, {}, use_auto_resize) {};

    std::map<std::string, ov::Tensor> Infer(const cv::Mat& frame) override;

    std::vector<cv::RotatedRect> postProcess(
        const std::map<std::string, ov::Tensor>& output_tensors, const cv::Size& image_size,
        const cv::Size& image_shape, float cls_conf_threshold, float link_conf_threshold);
private:
    cv::Mat decodeImageByJoin(
        const std::vector<float>& cls_data, const ov::Shape& cls_data_shape,
        const std::vector<float>& link_data, const ov::Shape& link_data_shape,
        float cls_conf_threshold, float link_conf_threshold);
};
