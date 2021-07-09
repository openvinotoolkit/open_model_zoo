/*
// Copyright (C) 2021 Intel Corporation
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

#include "utils/image_utils.h"

cv::Mat resizeImageExt(const cv::Mat& mat, int width, int height, RESIZE_MODE resizeMode, bool hqResize, cv::Rect* roi) {
    if (width == mat.cols && height == mat.rows) {
        return mat;
    }

    cv::Mat dst;
    int interpMode = hqResize ? cv::INTER_LINEAR : cv::INTER_CUBIC;

    switch (resizeMode) {
    case RESIZE_FILL:
    {
        cv::resize(mat, dst, cv::Size(width, height), interpMode);
        if (roi) {
            *roi = cv::Rect(0, 0, width, height);
        }
        break;
    }
    case RESIZE_KEEP_ASPECT:
    case RESIZE_KEEP_ASPECT_LETTERBOX:
    {
        double scale = std::min(static_cast<double>(width) / mat.cols, static_cast<double>(height) / mat.rows);
        cv::Mat resizedImage;
        cv::resize(mat, resizedImage, cv::Size(0, 0), scale, scale, interpMode);

        int dx = resizeMode == RESIZE_KEEP_ASPECT ? 0 : (width - resizedImage.cols) / 2;
        int dy = resizeMode == RESIZE_KEEP_ASPECT ? 0 : (height - resizedImage.rows) / 2;

        cv::copyMakeBorder(resizedImage, dst, dy, height - resizedImage.rows - dy,
            dx, width - resizedImage.cols - dx, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        if (roi) {
            *roi = cv::Rect(dx, dy, resizedImage.cols, resizedImage.rows);
        }
        break;
    }
    }
    return dst;
}
