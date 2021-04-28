// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <opencv2/videoio.hpp>
#include <opencv2/gapi/garg.hpp>

namespace custom {
class CustomCapSource : public cv::gapi::wip::IStreamSource
{
public:
    explicit CustomCapSource(const cv::VideoCapture& cap) : cap(cap) { prep(); }

protected:
    cv::VideoCapture cap;
    cv::Mat first;
    bool first_pulled = false;
    cv::Mat clear_frame;
    void prep() {
        GAPI_Assert(first.empty());
        cv::Mat tmp;
        if (!cap.read(tmp)) {
            GAPI_Assert(false && "Couldn't grab the frame");
        }
        first = tmp.clone();
    }

    virtual bool pull(cv::gapi::wip::Data &data) override {
        if (!first_pulled) {
            GAPI_Assert(!first.empty());
            first_pulled = true;
            data = first;
            return true;
        }
        if (!cap.isOpened()) return false;
        cv::Mat frame;
        if (!cap.read(frame)) {
            return false;
        }
        data = frame.clone();
        return true;
    }

    virtual cv::GMetaArg descr_of() const override {
        GAPI_Assert(!first.empty());
        return cv::GMetaArg{ cv::descr_of(first) };
    }
};

std::tuple<cv::VideoCapture, cv::Size, size_t> setInput(const std::string& input,
                                                        const cv::Size& camera_res,
                                                        const int limit) {
    cv::VideoCapture cap;
    try {
        // If stoi() throws exception input should be a path not a camera id
        cap = cv::VideoCapture(std::stoi(input));
    } catch (std::invalid_argument&) {
        slog::info << "Input source is treated as a file path" << slog::endl;
        cap = cv::VideoCapture(input);
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, camera_res.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, camera_res.height);
    const cv::Size frame_size(int(cap.get(cv::CAP_PROP_FRAME_WIDTH)), int(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    const int video_length = int(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const size_t num_frames = limit > 0 && (limit < video_length)
        ? limit
        : video_length;
    return std::make_tuple(cap, frame_size, num_frames);
}
} // namespace custom
