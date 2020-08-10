// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>

class ImagesCapture {
public:
    const bool loop;

    ImagesCapture(bool loop) : loop{loop} {}
    virtual double fps() const = 0;
    virtual size_t lastImageId() const = 0;
    virtual cv::Mat read() = 0;
    virtual ~ImagesCapture() = default;
};

// An advanced version of
// try {
//     return cv::VideoCapture(std::stoi(input));
// } catch (const std::invalid_argument&) {
//     return cv::VideoCapture(input);
// } catch (const std::out_of_range&) {
//     return cv::VideoCapture(input);
// }
std::unique_ptr<ImagesCapture> openImagesCapture(const std::string &input,
    bool loop, size_t initialImageId=0,  // Non camera options
    size_t readLengthLimit=std::numeric_limits<size_t>::max()  // General option
);
