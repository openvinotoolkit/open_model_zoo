// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/gapi/garray.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/gopaque.hpp>

namespace cv {
class GMat;
struct GMatDesc;
}  // namespace cv

namespace custom {
using GMat3 = std::tuple<cv::GMat, cv::GMat, cv::GMat>;
using GMats = cv::GArray<cv::GMat>;
using GRects = cv::GArray<cv::Rect>;
using GSize = cv::GOpaque<cv::Size>;

// clang-format off
G_API_OP(PrepareEyes,
         <std::tuple<GMats,
                     GMats>(cv::GMat, GRects, GRects, GMats, cv::Size)>,
         "custom.gapi.prepareEyes") {
    static std::tuple<cv::GArrayDesc, cv::GArrayDesc> outMeta(const cv::GMatDesc&,
                                                              const cv::GArrayDesc&,
                                                              const cv::GArrayDesc&,
                                                              const cv::GArrayDesc&,
                                                              const cv::Size&) {
        return std::make_tuple(cv::empty_array_desc(),
                               cv::empty_array_desc());
    }
};

G_API_OP(ParseSSD,
         <std::tuple<GRects, cv::GArray<float>>(cv::GMat, GSize, float)>,
         "custom.gaze_estimation.parseSSD") {
    static std::tuple<cv::GArrayDesc, cv::GArrayDesc> outMeta(const cv::GMatDesc&,
                                                              const cv::GOpaqueDesc&,
                                                              const float) {
        return std::make_tuple(cv::empty_array_desc(),
                               cv::empty_array_desc());
    }
};

G_API_OP(ProcessPoses,
         <std::tuple<cv::GArray<cv::Point3f>, GMats>(GMats, GMats, GMats)>,
         "custom.gaze_estimation.processPoses") {
    static std::tuple<cv::GArrayDesc, cv::GArrayDesc> outMeta(const cv::GArrayDesc&,
                                                              const cv::GArrayDesc&,
                                                              const cv::GArrayDesc&) {
        return std::make_tuple(cv::empty_array_desc(),
                               cv::empty_array_desc());
    }
};

G_API_OP(ProcessEyes,
         <std::tuple<cv::GArray<int>,
                     cv::GArray<int>>(cv::GMat, GMats, GMats)>,
         "custom.gaze_estimation.processEyes") {
    static std::tuple<cv::GArrayDesc,
                      cv::GArrayDesc> outMeta(const cv::GMatDesc&,
                                              const cv::GArrayDesc&,
                                              const cv::GArrayDesc&) {
        return std::make_tuple(cv::empty_array_desc(),
                               cv::empty_array_desc());
    }
};

G_API_OP(ProcessGazes,
         <cv::GArray<cv::Point3f>(GMats, GMats)>,
         "custom.gaze_estimation.processGazes") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
                                  const cv::GArrayDesc&) {
        return cv::empty_array_desc();
    }
};

G_API_OP(ProcessLandmarks,
         <std::tuple<GRects,
                     GRects,
                     cv::GArray<cv::Point2f>,
                     cv::GArray<cv::Point2f>,
                     cv::GArray<std::vector<cv::Point>>>(cv::GMat, GMats, GRects)>,
         "custom.gaze_estimation.processLandmarks") {
    static std::tuple<cv::GArrayDesc,
                      cv::GArrayDesc,
                      cv::GArrayDesc,
                      cv::GArrayDesc,
                      cv::GArrayDesc> outMeta(const cv::GMatDesc&,
                                              const cv::GArrayDesc&,
                                              const cv::GArrayDesc&) {
        return std::make_tuple(cv::empty_array_desc(),
                               cv::empty_array_desc(),
                               cv::empty_array_desc(),
                               cv::empty_array_desc(),
                               cv::empty_array_desc());
    }
};
// clang-format on
}  // namespace custom
