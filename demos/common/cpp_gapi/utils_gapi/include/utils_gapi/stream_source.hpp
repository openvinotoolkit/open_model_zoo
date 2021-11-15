// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utils/images_capture.h>
#include <opencv2/gapi.hpp>

namespace custom {
class CommonCapSrc : public cv::gapi::wip::IStreamSource
{
public:
    explicit CommonCapSrc(std::shared_ptr<ImagesCapture>& cap);

protected:
    std::shared_ptr<ImagesCapture> cap;
    cv::Mat first;
    bool first_pulled = false;

    void preparation();
    virtual bool pull(cv::gapi::wip::Data &data) override;
    virtual cv::GMetaArg descr_of() const override;
};

} // namespace custom
