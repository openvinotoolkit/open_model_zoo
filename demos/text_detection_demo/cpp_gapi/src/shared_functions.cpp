// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_functions.hpp"

size_t custom::getTopLeftPointIdx(const std::vector<cv::Point2f>& points) {
    cv::Point2f mostLeft(std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max());
    cv::Point2f secondLeft(std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max());
    size_t mostLeftIdx = -1;
    size_t secondLeftIdx = -1;
    for (size_t i = 0; i < points.size() ; i++) {
        if (mostLeft.x > points[i].x) {
            if (mostLeft.x < std::numeric_limits<float>::max()) {
                secondLeft = mostLeft;
                secondLeftIdx = mostLeftIdx;
            }
            mostLeft = points[i];
            mostLeftIdx = i;
        }
        if (secondLeft.x > points[i].x && points[i] != mostLeft) {
            secondLeft = points[i];
            secondLeftIdx = i;
        }
    }
    if (secondLeft.y < mostLeft.y) {
        mostLeft = secondLeft;
        mostLeftIdx = secondLeftIdx;
    }
    return mostLeftIdx;
}
