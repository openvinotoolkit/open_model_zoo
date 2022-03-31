// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <stddef.h>

#include <limits>
#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include "utils/performance_metrics.hpp"

enum class read_type { efficient, safe };

class ImagesCapture {
public:
    const bool loop;

    ImagesCapture(bool loop) : loop{loop} {}
    virtual double fps() const = 0;
    virtual cv::Mat read() = 0;
    virtual std::string getType() const = 0;
    const PerformanceMetrics& getMetrics() {
        return readerMetrics;
    }
    virtual ~ImagesCapture() = default;

protected:
    PerformanceMetrics readerMetrics;
};

// An advanced version of
// try {
//     return cv::VideoCapture(std::stoi(input));
// } catch (const std::invalid_argument&) {
//     return cv::VideoCapture(input);
// } catch (const std::out_of_range&) {
//     return cv::VideoCapture(input);
// }
// Some VideoCapture backends continue owning the video buffer under cv::Mat. safe_copy forses to return a copy from
// read()
// https://github.com/opencv/opencv/blob/46e1560678dba83d25d309d8fbce01c40f21b7be/modules/gapi/include/opencv2/gapi/streaming/cap.hpp#L72-L76
std::unique_ptr<ImagesCapture> openImagesCapture(
    const std::string& input,
    bool loop,
    read_type type = read_type::efficient,
    size_t initialImageId = 0,
    size_t readLengthLimit = std::numeric_limits<size_t>::max(),  // General option
    cv::Size cameraResolution = {1280, 720});
