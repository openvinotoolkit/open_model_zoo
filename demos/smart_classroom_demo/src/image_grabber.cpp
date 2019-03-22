// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <string>

#include "image_grabber.hpp"

ImageGrabber::ImageGrabber(const std::string& fname) {
    is_sequence = false;
    if (fname == "cam") {
        is_opened = cap.open(0);
    } else {
        is_opened = cap.open(fname);
    }
    cap_frame_index = -1;
    current_video_idx = 0;
    videos.push_back(fname);
    current_frame_idx = 0;
}

std::string ImageGrabber::GetVideoPath() const {
    return current_video_idx >= 0 ? videos[current_video_idx] : std::string("");
}

int ImageGrabber::GetFPS() const {
    return static_cast<int>(cap.get(cv::CAP_PROP_FPS));
}

bool ImageGrabber::IsOpened() const { return is_opened; }

int ImageGrabber::GetFrameIndex() const { return cap_frame_index; }

bool ImageGrabber::NextVideo() { return true; }

bool ImageGrabber::GrabNext() {
    cap_frame_index++;
    return cap.grab();
}

bool ImageGrabber::Retrieve(cv::Mat& img) { return cap.retrieve(img); }
