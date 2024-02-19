// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>
#include <string>
#include <map>
#include <vector>

#include "utils/common.hpp"
#include "utils/ocv_common.hpp"
#include "utils/slog.hpp"

namespace gaze_estimation {
class IEWrapper {
public:
    IEWrapper(ov::Core& core,
              const std::string& modelPath,
              const std::string& modelType,
              const std::string& deviceName);
    // For setting input tensors containing images
    void setInputTensor(const std::string& tensorName, const cv::Mat& image);
    // For setting input tensors containing vectors of data
    void setInputTensor(const std::string& tensorName, const std::vector<float>& data);

    // Get output tensor content as a vector given its name
    void getOutputTensor(const std::string& tensorName, std::vector<float>& output);

    const std::map<std::string, ov::Shape>& getInputTensorDimsInfo() const;
    const std::map<std::string, ov::Shape>& getOutputTensorDimsInfo() const;

    std::string expectSingleInput() const;
    std::string expectSingleOutput() const;

    void expectImageInput(const std::string& tensorName) const;

    void reshape(const std::map<std::string, ov::Shape>& newTensorsDimsInfo);

    void infer();

private:
    std::string modelPath;
    std::string modelType;
    std::string deviceName;
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    std::map<std::string, ov::Shape> input_tensors_dims_info;
    std::map<std::string, ov::Shape> output_tensors_dims_info;

    void setExecPart();
};
}  // namespace gaze_estimation
