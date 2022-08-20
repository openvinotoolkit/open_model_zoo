// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/gapi/gmetaarg.hpp>
#include <opencv2/gapi/streaming/source.hpp>
#include <opencv2/gapi/media.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

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

    explicit MediaBGRAdapter(cv::Mat m, Cb cb = [](){})
        : m_mat(m), m_cb(cb) {
    }

    cv::GFrameDesc meta() const override {
        return cv::GFrameDesc{cv::MediaFormat::BGR, cv::Size(m_mat.cols, m_mat.rows)};
    }

    cv::MediaFrame::View access(cv::MediaFrame::Access) override {
        cv::MediaFrame::View::Ptrs pp = { m_mat.ptr(), nullptr, nullptr, nullptr };
        cv::MediaFrame::View::Strides ss = { m_mat.step, 0u, 0u, 0u };
        return cv::MediaFrame::View(std::move(pp), std::move(ss), Cb{m_cb});
    }

private:
    cv::Mat m_mat;
    Cb m_cb;
};

class MediaCommonCapSrc : public cv::gapi::wip::IStreamSource {
public:
    explicit MediaCommonCapSrc(std::shared_ptr<ImagesCapture>& img_cap)
        : cap(std::make_shared<CommonCapSrc>(img_cap)) {
    }

    bool pull(cv::gapi::wip::Data& data) {
        if (cap->pull(data)) {
            data = cv::MediaFrame::Create<MediaBGRAdapter>(cv::util::get<cv::Mat>(data));
            return true;
        }
        return false;
    }

    cv::GMetaArg descr_of() const override {
        return cv::GMetaArg{cv::GFrameDesc{cv::MediaFormat::BGR,
                            cv::util::get<cv::GMatDesc>(cap->descr_of()).size}};
    }

private:
    std::shared_ptr<CommonCapSrc> cap;
};

}  // namespace custom
