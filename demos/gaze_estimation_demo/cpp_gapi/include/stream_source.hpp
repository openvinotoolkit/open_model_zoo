// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utils/images_capture.h>

namespace custom {
class CustomCapSource : public cv::gapi::wip::IStreamSource
{
public:
    explicit CustomCapSource(std::shared_ptr<ImagesCapture>& cap) : cap(cap) {
        prep();
    }

protected:
    std::shared_ptr<ImagesCapture> cap;
    cv::Mat first;
    bool first_pulled = false;
    cv::Mat clear_frame;
    void prep() {
        GAPI_Assert(first.empty());
        cv::Mat tmp = cap->read();
        if (!tmp.data) {
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
        cv::Mat frame = cap->read();
        if (!frame.data) {
            return false;;
        }
        data = frame.clone();
        return true;
    }

    virtual cv::GMetaArg descr_of() const override {
        GAPI_Assert(!first.empty());
        return cv::GMetaArg{ cv::descr_of(first) };
    }
};

} // namespace custom
