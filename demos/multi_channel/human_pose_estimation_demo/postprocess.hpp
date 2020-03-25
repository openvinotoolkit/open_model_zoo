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

size_t constexpr keypointsNumber = 18;

std::vector<HumanPose> postprocess(
        float const* heatMapsData, int const heatMapOffset, int const nHeatMaps,
        float const* pafsData, int const pafOffset, int const nPafs,
        int const featureMapWidth, int const featureMapHeight,
        cv::Size const& imageSize);
