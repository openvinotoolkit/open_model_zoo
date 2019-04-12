// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <memory>
#include <utility>
#include <string>
#include <opencv2/core.hpp>

using ImageWithFrameIndex = std::pair<cv::Mat, int>;

class ImageReader {
public:
    virtual bool IsOpened() const = 0;
    virtual void SetFrameIndex(size_t frame_index) = 0;
    virtual double GetFrameRate() const = 0;
    virtual int FrameIndex() const = 0;
    virtual ImageWithFrameIndex Read() = 0;

    virtual ~ImageReader() {}

    /// @brief Create ImageReader to read from a folder with images.
    static std::unique_ptr<ImageReader> CreateImageReaderForImageFolder(
        const std::string& folder_path, size_t start_frame_index = 1);

    /// @brief Create ImageReader to read from a video file.
    static std::unique_ptr<ImageReader> CreateImageReaderForVideoFile(
        const std::string& file_path);

    /// @brief Create ImageReader to read either from a video file
    ///        (if the path points to a file) or from a folder with images
    ///        (if the path points to a folder)
    static std::unique_ptr<ImageReader> CreateImageReaderForPath(
        const std::string& path);
};


