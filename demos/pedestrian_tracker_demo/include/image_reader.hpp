/*
// Copyright (c) 2018 Intel Corporation
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


