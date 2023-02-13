/*
// Copyright (C) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "utils/color_palette.hpp"


double DefaultColorPalette::getRandom(double a, double b) {
    static std::default_random_engine e;
    std::uniform_real_distribution<> dis(a, std::nextafter(b, std::numeric_limits<double>::max()));
    return dis(e);
}

double DefaultColorPalette::distance(const cv::Scalar& c1, const cv::Scalar& c2) {
    auto dh = std::fmin(std::fabs(c1[0] - c2[0]), 1 - fabs(c1[0] - c2[0])) * 2;
    auto ds = std::fabs(c1[1] - c2[1]);
    auto dv = std::fabs(c1[2] - c2[2]);

    return dh * dh + ds * ds + dv * dv;
}

cv::Scalar DefaultColorPalette::maxMinDistance(const std::vector<cv::Scalar>& colorSet,
                                     const std::vector<cv::Scalar>& colorCandidates) {
    std::vector<double> distances;
    distances.reserve(colorCandidates.size());
    for (auto& c1 : colorCandidates) {
        auto min =
            *std::min_element(colorSet.begin(), colorSet.end(), [&c1](const cv::Scalar& a, const cv::Scalar& b) {
                return distance(c1, a) < distance(c1, b);
            });
        distances.push_back(distance(c1, min));
    }
    auto max = std::max_element(distances.begin(), distances.end());
    return colorCandidates[std::distance(distances.begin(), max)];
}

cv::Scalar DefaultColorPalette::hsv2rgb(const cv::Scalar& hsvColor) {
    cv::Mat rgb;
    cv::Mat hsv(1, 1, CV_8UC3, hsvColor);
    cv::cvtColor(hsv, rgb, cv::COLOR_HSV2RGB);
    return cv::Scalar(rgb.data[0], rgb.data[1], rgb.data[2]);
}

DefaultColorPalette::DefaultColorPalette(size_t n) {
    palette.reserve(n);
    std::vector<cv::Scalar> hsvColors(1, {1., 1., 1.});
    std::vector<cv::Scalar> colorCandidates;
    size_t numCandidates = 100;

    hsvColors.reserve(n);
    colorCandidates.resize(numCandidates);
    for (size_t i = 1; i < n; ++i) {
        std::generate(colorCandidates.begin(), colorCandidates.end(), []() {
            return cv::Scalar{getRandom(), getRandom(0.8, 1.0), getRandom(0.5, 1.0)};
        });
        hsvColors.push_back(maxMinDistance(hsvColors, colorCandidates));
    }

    for (auto& hsv : hsvColors) {
        // Convert to OpenCV HSV format
        hsv[0] *= 179;
        hsv[1] *= 255;
        hsv[2] *= 255;

        palette.push_back(hsv2rgb(hsv));
    }
}

const cv::Scalar& DefaultColorPalette::operator[](size_t index) const {
    return palette[index % palette.size()];
}