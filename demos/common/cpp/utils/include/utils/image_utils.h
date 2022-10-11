/*
// Copyright (C) 2021-2022 Intel Corporation
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

#include <opencv2/opencv.hpp>

enum RESIZE_MODE {
    RESIZE_FILL,
    RESIZE_KEEP_ASPECT,
    RESIZE_KEEP_ASPECT_LETTERBOX
};

enum INTERPOLATION_MODE {
    NEAREST,
    LINEAR,
    CUBIC,
    AREA
};

cv::Mat resizeImageExt(const cv::Mat& mat, int width, int height, RESIZE_MODE resizeMode = RESIZE_FILL,
                       INTERPOLATION_MODE interpolationMode = LINEAR, cv::Rect* roi = nullptr);
