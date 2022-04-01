// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/infer/ie.hpp>

#include <inference_engine.hpp>

#include <utils/slog.hpp>

namespace IE = InferenceEngine;

namespace custom {

struct DetectedObject : public cv::Rect2f
{
    unsigned int labelID;
    std::string label;
    float confidence;
};

using GDetections = cv::GArray<DetectedObject>;
using GLabels = cv::GArray<std::string>;

G_API_OP(GYOLOv4TinyPostProcessingKernel, < GDetections(cv::GMat, cv::GMat, cv::GMat, GLabels, float, float, bool) >, "custom.yolov4_tiny_post_processing") {
        static cv::GArrayDesc outMeta(const cv::GMatDesc&,
                                      const cv::GMatDesc&,
                                      const cv::GMatDesc&,
                                      const cv::GArrayDesc&, const float, const float, const bool) {
            return cv::empty_array_desc();
        }
};

G_API_OP(GSmartFramingKernel, <cv::GMat(cv::GMat, GDetections)>, "custom.smart_framing") {
        static cv::GMatDesc outMeta(const cv::GMatDesc & in, const cv::GArrayDesc&) {
            return in;
        }
};

G_API_OP(GSuperResolutionPostProcessingKernel, < cv::GMat(cv::GMat) >, "custom.super_resolution_post_processing") {
        static cv::GMatDesc outMeta(const cv::GMatDesc & in) {
            cv::GMatDesc out_desc(CV_8U /* depth */, in.dims[1] /* channels */, cv::Size(in.dims[3], in.dims[2]), false /* planar */);
            return out_desc;
        }
};

G_API_OP(GCvt32Fto8U, <cv::GMat(cv::GMat)>, "custom.convertFP32ToU8") {
    static cv::GMatDesc outMeta(const cv::GMatDesc & in) {
        // NB: Input is ND mat.
        return cv::GMatDesc{ CV_8U, in.dims[1], cv::Size(in.dims[3], in.dims[2]) };
    }
};

cv::gapi::GKernelPackage kernels();

} // namespace custom
