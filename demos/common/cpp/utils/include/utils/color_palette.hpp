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

#include <stddef.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include <gflags/gflags.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

class DefaultColorPalette {
    private:
        std::vector<cv::Scalar> palette;

        static double getRandom(double a = 0.0, double b = 1.0);

        static double distance(const cv::Scalar& c1, const cv::Scalar& c2);

        static cv::Scalar maxMinDistance(const std::vector<cv::Scalar>& colorSet,
                                        const std::vector<cv::Scalar>& colorCandidates);

        static cv::Scalar hsv2rgb(const cv::Scalar& hsvColor);

    public:
        explicit DefaultColorPalette(size_t n);

        const cv::Scalar& operator[](size_t index) const;
};