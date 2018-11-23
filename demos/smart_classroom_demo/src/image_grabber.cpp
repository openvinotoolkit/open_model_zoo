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
