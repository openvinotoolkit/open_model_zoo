// Copyright (C) 2021-2024 Intel Corporation
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
    } else {
        cv::Mat frame = cap->read();
        if (!frame.data) {
            return false;
        }
        data = frame.clone();
    }
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    data.meta[cv::gapi::streaming::meta_tag::timestamp] =
            int64_t{std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count()};
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
        auto &&original_meta = std::move(data.meta);
        data = cv::MediaFrame::Create<MediaBGRAdapter>(cv::util::get<cv::Mat>(data));
        data.meta = std::move(original_meta);
        return true;
    }
    return false;
}

cv::GMetaArg MediaCommonCapSrc::descr_of() const {
    return cv::GMetaArg{cv::GFrameDesc{cv::MediaFormat::BGR,
                                       cv::util::get<cv::GMatDesc>(CommonCapSrc::descr_of()).size}};
}
}  // namespace custom

namespace util {
cv::gapi::wip::onevpl::CfgParam createFromString(const std::string &line) {
    using namespace cv::gapi::wip;

    if (line.empty()) {
        throw std::runtime_error("Cannot parse CfgParam from emply line");
    }

    std::string::size_type name_endline_pos = line.find(':');
    if (name_endline_pos == std::string::npos) {
        throw std::runtime_error("Cannot parse CfgParam from: " + line +
                                 "\nExpected separator \":\"");
    }

    std::string name = line.substr(0, name_endline_pos);
    std::string value = line.substr(name_endline_pos + 1);

    return cv::gapi::wip::onevpl::CfgParam::create(name, value,
                                                   /* vpp params strongly optional */
                                                   name.find("vpp.") == std::string::npos);
}

std::vector<cv::gapi::wip::onevpl::CfgParam> parseVPLParams(const std::string& cfg_params) {
    std::vector<cv::gapi::wip::onevpl::CfgParam> source_cfgs;
    std::stringstream params_list(cfg_params);
    std::string line;
    while (std::getline(params_list, line, ',')) {
        source_cfgs.push_back(createFromString(line));
    }
    return source_cfgs;
}
} // namespace util
