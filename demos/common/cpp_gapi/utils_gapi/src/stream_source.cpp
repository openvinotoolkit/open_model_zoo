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

MediaBGRAdapter::MediaBGRAdapter(cv::Mat m, MediaBGRAdapter::Cb cb)
    : m_mat(m), m_cb(cb) {
}

cv::GFrameDesc MediaBGRAdapter::meta() const {
    return cv::GFrameDesc{cv::MediaFormat::BGR, m_mat.size()};
}

cv::MediaFrame::View MediaBGRAdapter::access(cv::MediaFrame::Access) {
    cv::MediaFrame::View::Ptrs pp = { m_mat.ptr(), nullptr, nullptr, nullptr };
    cv::MediaFrame::View::Strides ss = { m_mat.step, 0u, 0u, 0u };
    return cv::MediaFrame::View(std::move(pp), std::move(ss), MediaBGRAdapter::Cb{m_cb});
}

bool MediaCommonCapSrc::pull(cv::gapi::wip::Data& data) {
    if (CommonCapSrc::pull(data)) {
        data = cv::MediaFrame::Create<MediaBGRAdapter>(cv::util::get<cv::Mat>(data));
        return true;
    }
    return false;
}

cv::GMetaArg MediaCommonCapSrc::descr_of() const {
    return cv::GMetaArg{cv::GFrameDesc{cv::MediaFormat::BGR,
                                       cv::util::get<cv::GMatDesc>(CommonCapSrc::descr_of()).size}};
}
}  // namespace custom
