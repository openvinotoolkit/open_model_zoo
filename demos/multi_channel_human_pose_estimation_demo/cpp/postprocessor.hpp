/*
// Copyright (C) 2018-2019 Intel Corporation
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

#pragma once

#include <vector>

#include <opencv2/core/core.hpp>

#include "human_pose.hpp"

class Postprocessor {
public:
    explicit Postprocessor(int const upsampleRatio = 4, int const stride = 8, cv::Vec4i const pad = cv::Vec4i::all(0));
    void resizeFeatureMaps(std::vector<cv::Mat>& featureMaps) const;
    void correctCoordinates(std::vector<HumanPose>& poses,
                            const cv::Size& featureMapsSize,
                            const cv::Size& imageSize) const;
    ~Postprocessor() = default;
private:
    int upsampleRatio;
    int stride;
    cv::Vec4i pad;
};
