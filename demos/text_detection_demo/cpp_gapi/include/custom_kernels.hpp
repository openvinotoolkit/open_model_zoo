// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/infer.hpp>

using CompositeDecInputDescrs = std::pair<cv::GMatDesc, cv::GMatDesc>;
namespace cv { namespace detail {
    template<> struct CompileArgTag<CompositeDecInputDescrs> {
        static const char* tag() {
            return "custom.compositeDecoderInputDescrs";
        }
    };
}} // namespace cv::detail

namespace custom {
using GSize = cv::GOpaque<cv::Size>;
using GRRects = cv::GArray<cv::RotatedRect>;
using GMats = cv::GArray<cv::GMat>;
using GRRPoints = cv::GArray<std::vector<cv::Point2f>>;

G_API_OP(DetectionPostProcess, <GRRects(cv::GMat,cv::GMat,GSize,cv::Size,float,float,size_t)>,
         "sample.custom.text.detPostProc") {
    static cv::GArrayDesc outMeta(const cv::GMatDesc&, const cv::GMatDesc&, const cv::GOpaqueDesc&,
                                  const cv::Size&, float, float, size_t) {
        return cv::empty_array_desc();
    }
};

G_API_OP(PointsFromRRects, <GRRPoints(GRRects,GSize,bool)>, "sample.custom.text.ptsFromRrs") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc&, const cv::GOpaqueDesc&, const bool) {
        return cv::empty_array_desc();
    }
};

G_API_OP(CropLabels, <std::tuple<GMats,GRRPoints>(cv::GMat,GRRects,std::vector<size_t>,bool)>,
         "sample.custom.text.crop") {
    static std::tuple<cv::GArrayDesc, cv::GArrayDesc> outMeta(const cv::GMatDesc&,
                                                              const cv::GArrayDesc&,
                                                              const std::vector<size_t>&,
                                                              const bool) {
        return std::make_tuple(cv::empty_array_desc(), cv::empty_array_desc());
    }
};

G_API_OP(CompositeTRDecode, <GMats(GMats,GMats,size_t,size_t)>,
         "sample.custom.text.recogCompositeDecode") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc&, const cv::GArrayDesc&,
                                  const size_t, const size_t) {
        return cv::empty_array_desc();
    }
};

cv::gapi::GKernelPackage kernels();
} // namespace custom
