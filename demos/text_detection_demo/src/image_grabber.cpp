// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "image_grabber.hpp"

#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

Grabber::~Grabber() {}

VideoGrabber::VideoGrabber(const std::string &path, bool is_web_cam): cap(path) {
    if (!cap.isOpened()) throw std::runtime_error("Could not open a video: " + path);
    if (is_web_cam) {
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap.set(cv::CAP_PROP_AUTOFOCUS, 1);
    }
}

void VideoGrabber::GrabNextImage(cv::Mat *frame) {
    cap >> *frame;
}

ImageGrabber::ImageGrabber(const std::string &path) {
    image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) throw std::runtime_error("Could not read an image: " + path);
}

void ImageGrabber::GrabNextImage(cv::Mat *frame) {
    *frame = image.clone();
    image = cv::Mat();
}

ImageListGrabber::ImageListGrabber(const std::string &path) {
    std::ifstream fs(path);
    if (!fs.is_open()) throw std::runtime_error("Could not find an image list: " + path);
    while (!fs.eof()) {
        std::string str = "";
        std::getline(fs, str);

        std::istringstream iss(str);
        std::vector<std::string> results((std::istream_iterator<std::string>(iss)),
                                         std::istream_iterator<std::string>());

        if (!results.empty()) {
            image_paths.emplace_back(results[0]);
        }
    }
    index = 0;
}

void ImageListGrabber::GrabNextImage(cv::Mat *frame) {
  if (index >= image_paths.size()) {
      *frame = cv::Mat();
  } else {
      *frame = cv::imread(image_paths[index], cv::IMREAD_COLOR);
      if (frame->empty()) throw std::runtime_error("Could not read an image: " + image_paths[index]);
      index++;
  }
}

std::unique_ptr<Grabber> Grabber::make_grabber(const std::string& type, const std::string& path) {
    if (type == "image") {
        return std::unique_ptr<ImageGrabber>(new ImageGrabber(path));
    } else if (type == "list") {
        return std::unique_ptr<ImageListGrabber>(new ImageListGrabber(path));
    } else if (type == "video") {
        return std::unique_ptr<VideoGrabber>(new VideoGrabber(path));
    } else if (type == "webcam") {
        return std::unique_ptr<VideoGrabber>(new VideoGrabber(path, true));
    } else {
        throw std::runtime_error("Unknown data input type:" + type + ". Possible variants are \"image\", \"list\", \"video\", \"webcam\".");
    }
}
