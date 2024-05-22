// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detector.hpp"

#include <algorithm>

#define SSD_EMPTY_DETECTIONS_INDICATOR -1.0

using namespace detection;

namespace {
cv::Rect TruncateToValidRect(const cv::Rect& rect, const cv::Size& size) {
    auto tl = rect.tl(), br = rect.br();
    tl.x = std::max(0, std::min(size.width - 1, tl.x));
    tl.y = std::max(0, std::min(size.height - 1, tl.y));
    br.x = std::max(0, std::min(size.width, br.x));
    br.y = std::max(0, std::min(size.height, br.y));
    const int w = std::max(0, br.x - tl.x);
    const int h = std::max(0, br.y - tl.y);
    return cv::Rect(tl.x, tl.y, w, h);
}

cv::Rect IncreaseRect(const cv::Rect& r, float coeff_x, float coeff_y) {
    const cv::Point2f tl = r.tl();
    const cv::Point2f br = r.br();
    const cv::Point2f c = (tl * 0.5f) + (br * 0.5f);
    const cv::Point2f diff = c - tl;
    const cv::Point2f new_diff{diff.x * coeff_x, diff.y * coeff_y};
    const cv::Point2f new_tl = c - new_diff;
    const cv::Point2f new_br = c + new_diff;

    const cv::Point new_tl_int{static_cast<int>(std::floor(new_tl.x)), static_cast<int>(std::floor(new_tl.y))};
    const cv::Point new_br_int{static_cast<int>(std::ceil(new_br.x)), static_cast<int>(std::ceil(new_br.y))};

    return cv::Rect(new_tl_int, new_br_int);
}
}  // namespace

void FaceDetection::truncateRois(const cv::Mat& in,
                                 const std::vector<cv::Rect>& face_rois,
                                 std::vector<cv::Rect>& valid_face_rois) {
    for (const auto& roi : face_rois) {
        valid_face_rois.emplace_back(
            TruncateToValidRect(IncreaseRect(roi, config_.increase_scale_x, config_.increase_scale_y),
                                cv::Size(static_cast<int>(in.cols), static_cast<int>(in.rows))));
    }
}
