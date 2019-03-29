// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

class Grabber {
  public:
    virtual void GrabNextImage(cv::Mat *frame) = 0;
    static std::unique_ptr<Grabber> make_grabber(const std::string& type, const std::string& path);

    virtual ~Grabber();
};

class VideoGrabber : public Grabber {
  public:
    explicit VideoGrabber(const std::string &path, bool is_web_cam = false);
    virtual ~VideoGrabber() {}
    virtual void GrabNextImage(cv::Mat *frame);

  private:
    cv::VideoCapture cap;
};

class ImageGrabber : public Grabber {
  public:
    explicit ImageGrabber(const std::string &path);
    virtual ~ImageGrabber() {}
    virtual void GrabNextImage(cv::Mat *frame);

  private:
    cv::Mat image;
};

class ImageListGrabber : public Grabber {
  public:
    explicit ImageListGrabber(const std::string &path);
    virtual ~ImageListGrabber() {}

    virtual void GrabNextImage(cv::Mat *frame);

  private:
    std::vector<std::string> image_paths;
    size_t index;
};


