// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdio>
#include <string>

#define _USE_MATH_DEFINES
#include <cmath>

#include <utility>
#include <memory>
#include <map>
#include <vector>
#include <set>

#include "utils.hpp"

using namespace InferenceEngine;

namespace gaze_estimation {
void initializeIEObject(InferenceEngine::Core& ie,
                        const std::vector<std::pair<std::string, std::string>>& cmdOptions) {
    std::set<std::string> loadedDevices;
    for (auto && option : cmdOptions) {
        auto deviceName = option.first;
        auto networkName = option.second;

        if (deviceName.empty() || networkName.empty()) {
            continue;
        }

        if (loadedDevices.find(deviceName) != loadedDevices.end()) {
            continue;
        }
        slog::info << "Loading device " << deviceName << slog::endl;
        std::cout << ie.GetVersions(deviceName) << std::endl;

        /** Loading extensions for the CPU device **/
        if ((deviceName.find("CPU") != std::string::npos)) {
            loadedDevices.insert(deviceName);
        }
    }
}

void gazeVectorToGazeAngles(const cv::Point3f& gazeVector, cv::Point2f& gazeAngles) {
    auto r = cv::norm(gazeVector);

    double v0 = static_cast<double>(gazeVector.x);
    double v1 = static_cast<double>(gazeVector.y);
    double v2 = static_cast<double>(gazeVector.z);

    gazeAngles.x = static_cast<float>(180.0 / M_PI * (M_PI_2 + std::atan2(v2, v0)));
    gazeAngles.y = static_cast<float>(180.0 / M_PI * (M_PI_2 - std::acos(v1 / r)));
}

void putTimingInfoOnFrame(cv::Mat& image, double overallTime, double inferenceTime) {
    auto frameHeight = image.rows;
    double fontScale = 1.6 * frameHeight / 640;
    auto fontColor = cv::Scalar(0, 0, 255);
    int thickness = 2;

    double overallFPS = 1000. / overallTime;
    double inferenceFPS = 1000. / inferenceTime;

    cv::putText(image,
                cv::format("Overall FPS: %0.0f, Inference FPS: %0.0f", overallFPS, inferenceFPS),
                cv::Point(10, static_cast<int>(30 * fontScale / 1.6)), cv::FONT_HERSHEY_PLAIN, fontScale, fontColor, thickness);
}
}  // namespace gaze_estimation
