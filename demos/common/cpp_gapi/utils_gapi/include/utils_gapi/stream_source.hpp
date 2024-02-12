// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/gapi/gmetaarg.hpp>
#include <opencv2/gapi/streaming/source.hpp>
#include <opencv2/gapi/media.hpp>
#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/gapi/streaming/onevpl/source.hpp>

class ImagesCapture;
namespace cv {
namespace gapi {
namespace wip {
struct Data;
}  // namespace wip
}  // namespace gapi
}  // namespace cv

namespace custom {
class CommonCapSrc : public cv::gapi::wip::IStreamSource {
public:
    explicit CommonCapSrc(std::shared_ptr<ImagesCapture>& cap);

public:
    std::shared_ptr<ImagesCapture> cap;
    cv::Mat first;
    bool first_pulled = false;

    void preparation();
    bool pull(cv::gapi::wip::Data& data) override;
    cv::GMetaArg descr_of() const override;
};

class MediaBGRAdapter final: public cv::MediaFrame::IAdapter {
public:
    using Cb = cv::MediaFrame::View::Callback;

    explicit MediaBGRAdapter(cv::Mat m, Cb cb = [](){});

    cv::GFrameDesc meta() const override;
    cv::MediaFrame::View access(cv::MediaFrame::Access) override;

private:
    cv::Mat m_mat;
    Cb m_cb;
};

class MediaCommonCapSrc : public CommonCapSrc {
    using CommonCapSrc::CommonCapSrc;

    bool pull(cv::gapi::wip::Data& data);
    cv::GMetaArg descr_of() const override;
};

}  // namespace custom

namespace util {
cv::gapi::wip::onevpl::CfgParam createFromString(const std::string &line);
std::vector<cv::gapi::wip::onevpl::CfgParam> parseVPLParams(const std::string& cfg_params);
} // namespace util
