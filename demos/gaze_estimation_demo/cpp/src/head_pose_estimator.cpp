// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "head_pose_estimator.hpp"

namespace gaze_estimation {

const std::pair<const char*, float cv::Point3f::*> OUTPUTS[] = {
    {"angle_y_fc", &cv::Point3f::x},
    {"angle_p_fc", &cv::Point3f::y},
    {"angle_r_fc", &cv::Point3f::z},
};

HeadPoseEstimator::HeadPoseEstimator(
    ov::Core& ie, const std::string& modelPath, const std::string& deviceName) :
        ieWrapper(ie, modelPath, modelType, deviceName)
{
    inputTensorName = ieWrapper.expectSingleInput();
    ieWrapper.expectImageInput(inputTensorName);

    const auto& outputInfo = ieWrapper.getOutputTensorDimsInfo();

    for (const auto& output: OUTPUTS) {
        auto it = outputInfo.find(output.first);

        if (it == outputInfo.end())
            throw std::runtime_error(
                modelPath + ": expected to have output named \"" + output.first + "\"");

        bool correctDims = std::all_of(it->second.begin(), it->second.end(),
            [](unsigned long n) { return n == 1; });
        if (!correctDims)
            throw std::runtime_error(
                modelPath + ": expected \"" + output.first + "\" to have total size 1");
    }
}

void HeadPoseEstimator::estimate(const cv::Mat& image, FaceInferenceResults& outputResults) {
    auto faceBoundingBox = outputResults.faceBoundingBox;
    auto faceCrop(cv::Mat(image, faceBoundingBox));

    ieWrapper.setInputTensor(inputTensorName, faceCrop);
    ieWrapper.infer();

    std::vector<float> outputValue;

    for (const auto &output: OUTPUTS) {
        ieWrapper.getOutputTensor(output.first, outputValue);
        outputResults.headPoseAngles.*output.second = outputValue[0];
    }
}

HeadPoseEstimator::~HeadPoseEstimator() {
}
}  // namespace gaze_estimation
