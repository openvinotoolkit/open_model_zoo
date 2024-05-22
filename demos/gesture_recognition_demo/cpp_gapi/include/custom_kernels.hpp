// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/gapi/garray.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/gopaque.hpp>

struct TrackedObject;

// clang-format off
namespace custom {
G_API_OP(GetFastFrame,
         <cv::GMat(cv::GArray<cv::GMat>, cv::Size)>, "custom.get_fast_frame") {
    static cv::GMatDesc outMeta(const cv::GArrayDesc &in,
                                const cv::Size& frame_size) {
        return cv::GMatDesc{CV_8U, 3, frame_size};
    }
};

G_API_OP(ExtractBoundingBox,
         <cv::GArray<TrackedObject>(cv::GMat,
                                    cv::GMat,
                                    cv::Scalar)>,
         "custom.bb_extract") {
    static cv::GArrayDesc outMeta(const cv::GMatDesc &in,
                                  const cv::GMatDesc&,
                                  const cv::Scalar) {
        return cv::empty_array_desc();
    }
};

G_API_OP(TrackPerson,
         <cv::GArray<TrackedObject>(cv::GMat,
                                    cv::GArray<TrackedObject>)>,
         "custom.track") {
    static cv::GArrayDesc outMeta(const cv::GMatDesc &in,
                                  const cv::GArrayDesc&) {
        return cv::empty_array_desc();
    }
};

G_API_OP(ConstructClip,
         <cv::GArray<cv::GMat>(const cv::GArray<cv::GMat>,
                               const cv::GArray<TrackedObject>,
                               const cv::Scalar,
                               const cv::Size,
                               const cv::GOpaque<std::shared_ptr<size_t>>)>,
         "custom.construct_clip") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
                                  const cv::GArrayDesc&,
                                  const cv::Scalar&,
                                  const cv::Size&,
                                  const cv::GOpaqueDesc&) {
        return cv::empty_array_desc();
    }
};

G_API_OP(GestureRecognitionPostprocessing,
         <cv::GOpaque<int>(cv::GArray<cv::GMat>,
                           float)>,
         "custom.ar_postproc") {
    static cv::GOpaqueDesc outMeta(const cv::GArrayDesc&,
                                   const float) {
        return cv::empty_gopaque_desc();
    }
};

cv::gapi::GKernelPackage kernels();
// clang-format on
}  // namespace custom
