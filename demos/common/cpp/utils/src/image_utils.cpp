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
    {
        double scale = std::min(static_cast<double>(width) / mat.cols, static_cast<double>(height) / mat.rows);
        int newW = static_cast<int>(mat.cols * scale);
        int newH = static_cast<int>(mat.rows * scale);
        cv::Mat resizedImage;
        cv::resize(mat, resizedImage, cv::Size(0, 0), scale, scale, interpMode);
        cv::copyMakeBorder(resizedImage, dst, 0, height - newH,
            0, width - newW, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        if (roi) {
            *roi = cv::Rect(0, 0, newW, newH);
        }
        break;
    }
    case RESIZE_KEEP_ASPECT_LETTERBOX:
    {
        double scale = std::min(static_cast<double>(width) / mat.cols, static_cast<double>(height) / mat.rows);
        int newW = static_cast<int>(mat.cols * scale);
        int newH = static_cast<int>(mat.rows * scale);
        cv::Mat resizedImage;
        int dx = (width - newW) / 2;
        int dy = (height - newH) / 2;
        cv::resize(mat, resizedImage, cv::Size(0, 0), scale, scale, interpMode);
        cv::copyMakeBorder(resizedImage, dst, dy, dy,
            dx, dx, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        if (roi) {
            *roi = cv::Rect(dx, dy, newW, newH);
        }
        break;
    }
    }
    return dst;
}
