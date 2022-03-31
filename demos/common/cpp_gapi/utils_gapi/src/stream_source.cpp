// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils_gapi/stream_source.hpp"

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/own/assert.hpp>

#include <utils/images_capture.h>

namespace custom {
CommonCapSrc::CommonCapSrc(std::shared_ptr<ImagesCapture>& imagesCapture) : cap(imagesCapture) {
    preparation();
}

void CommonCapSrc::preparation() {
    GAPI_Assert(first.empty());
    cv::Mat tmp = cap->read();
    if (!tmp.data) {
        GAPI_Assert(false && "Couldn't grab the first frame");
    }
    first = tmp.clone();
}

bool CommonCapSrc::pull(cv::gapi::wip::Data& data) {
    if (!first_pulled) {
        GAPI_Assert(!first.empty());
        first_pulled = true;
        data = first;
        return true;
    }
    cv::Mat frame = cap->read();
    if (!frame.data) {
        return false;
    }
    data = frame.clone();
    return true;
}

cv::GMetaArg CommonCapSrc::descr_of() const {
    GAPI_Assert(!first.empty());
    return cv::GMetaArg{cv::descr_of(first)};
}
}  // namespace custom
