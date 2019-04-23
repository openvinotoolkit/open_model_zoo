// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "image_reader.hpp"
#include <details/ie_exception.hpp>
#include <iomanip>
#include <string>
#include <memory>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

namespace {
bool IsFolder(const std::string& folder_path) {
    struct stat folder_info;
    if ( stat( folder_path.c_str(), &folder_info ) != 0 )
        return false;
    if ( folder_info.st_mode & S_IFDIR )
        return true;
    return false;
}
bool IsFile(const std::string& path) {
    struct stat info;
    if ( stat( path.c_str(), &info ) != 0 )
        return false;
    if ( info.st_mode & S_IFREG )
        return true;
    return false;
}
}  // anonymous namespace

class ImageReaderForFolder: public ImageReader {
public:
    ImageReaderForFolder(const std::string& folder_path, size_t start_frame_index) {
        folder_path_ = folder_path;
        frame_index_ = start_frame_index;
    }

    bool IsOpened() const {
        return IsFolder(folder_path_);
    }
    void SetFrameIndex(size_t frame_index) {
        frame_index_ = frame_index;
    }

    int FrameIndex() const {
        return frame_index_;
    }

    ImageWithFrameIndex Read() {
        auto path = GetImagePath(folder_path_, frame_index_);
        cv::Mat img = cv::imread(path);

        ImageWithFrameIndex result;
        result.first = img;
        result.second = frame_index_;

        frame_index_++;
        return result;
    }

    // Note that for images folder
    // the default frame rate for DukeMTMC dataset is returned
    double GetFrameRate() const {return 60.0;}

private:
    std::string folder_path_;
    size_t frame_index_ = 1;

    static std::string GetImagePath(const std::string& folder_path,
                                    size_t frame_index) {
        std::stringstream strstr;
        strstr << folder_path << "/"
            << std::internal
            << std::setfill('0')
            << std::setw(10)
            << frame_index
            << ".jpg";
        return strstr.str();
    }
};

class ImageReaderForVideoFile: public ImageReader {
public:
    explicit ImageReaderForVideoFile(const std::string& file_path)
        : video_capture(file_path) {}

    bool IsOpened() const {
        return video_capture.isOpened();
    }
    void SetFrameIndex(size_t frame_index) {
        THROW_IE_EXCEPTION << "ImageReader does not set frame index in video, "
            << "since in the current implementation it is not precise";
    }

    int FrameIndex() const {
        return frame_index_;
    }

    ImageWithFrameIndex Read() {
        ImageWithFrameIndex result;
        video_capture >> result.first;
        result.second = frame_index_;
        frame_index_++;
        return result;
    }

    double GetFrameRate() const {
        double video_fps = video_capture.get(cv::CAP_PROP_FPS);
        if ((video_fps <= 0) || (video_fps > 200)) {
            video_fps = 30;
        }
        return video_fps;
    }

private:
    size_t frame_index_ = 1;
    cv::VideoCapture video_capture;
};

std::unique_ptr<ImageReader> ImageReader::CreateImageReaderForImageFolder(
    const std::string& folder_path, size_t start_frame_index) {
    return std::unique_ptr<ImageReader>(
        new ImageReaderForFolder(folder_path, start_frame_index));
}

std::unique_ptr<ImageReader> ImageReader::CreateImageReaderForVideoFile(
    const std::string& file_path) {
    return std::unique_ptr<ImageReader>(
        new ImageReaderForVideoFile(file_path));
}

std::unique_ptr<ImageReader> ImageReader::CreateImageReaderForPath(
    const std::string& path) {
    if (IsFolder(path))
        return ImageReader::CreateImageReaderForImageFolder(path);

    if (IsFile(path))
        return ImageReader::CreateImageReaderForVideoFile(path);

    return std::unique_ptr<ImageReader>();
}
