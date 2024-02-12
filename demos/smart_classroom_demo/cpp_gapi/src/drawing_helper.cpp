// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "drawing_helper.hpp"

#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "tracker.hpp"


DrawingHelper::DrawingHelper(bool show, int num_top_persons) : no_show_(show), num_top_persons_(num_top_persons) {
    if (!no_show_) {
        cv::namedWindow(main_window_name_);
    }
    if (!no_show_ && num_top_persons_ > 0) {
        cv::namedWindow(top_window_name_);
        CreateTopWindow();
        ClearTopWindow();
    }
}

cv::Size DrawingHelper::GetOutputSize(const cv::Size& input_size) {
    if (input_size.width > max_input_width_) {
        float ratio = static_cast<float>(input_size.height) / input_size.width;
        return cv::Size(max_input_width_, cvRound(ratio * max_input_width_));
    }
    return input_size;
}

float DrawingHelper::CalculateIoM(const cv::Rect& rect1, const cv::Rect& rect2) {
    const int area1 = rect1.area();
    const int area2 = rect2.area();
    const float area_min = static_cast<float>(std::min(area1, area2));
    const float area_intersect = static_cast<float>((rect1 & rect2).area());
    return area_intersect / area_min;
}

cv::Rect DrawingHelper::DecreaseRectByRelBorders(const cv::Rect& r) {
    const float w = static_cast<float>(r.width);
    const float h = static_cast<float>(r.height);
    const float left = std::ceil(w * 0.0f);
    const float top = std::ceil(h * 0.0f);
    const float right = std::ceil(w * 0.0f);
    const float bottom = std::ceil(h * .7f);
    return cv::Rect(r.x + static_cast<int>(left),
                    r.y + static_cast<int>(top),
                    static_cast<int>(r.width - left - right),
                    static_cast<int>(r.height - top - bottom));
}

int DrawingHelper::GetIndexOfTheNearestPerson(const TrackedObject& face,
                                              const std::vector<TrackedObject>& tracked_persons) {
    int argmax = -1;
    float max_iom = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < tracked_persons.size(); i++) {
        float iom = CalculateIoM(face.rect, DecreaseRectByRelBorders(tracked_persons[i].rect));
        if ((iom > 0) && (iom > max_iom)) {
            max_iom = iom;
            argmax = i;
        }
    }
    return argmax;
}

std::string DrawingHelper::GetActionTextLabel(const unsigned label, const std::vector<std::string>& actions_map) {
    if (label < actions_map.size()) {
        return actions_map[label];
    }
    return "__undefined__";
}

void DrawingHelper::GetNewFrameSize(const cv::Size& frame_size) {
    rect_scale_x_ = 1;
    rect_scale_y_ = 1;
    cv::Size new_size = GetOutputSize(frame_size);
    if (new_size != frame_size) {
        rect_scale_x_ = static_cast<float>(new_size.height) / frame_size.height;
        rect_scale_y_ = static_cast<float>(new_size.width) / frame_size.width;
    }
}

void DrawingHelper::CreateTopWindow() {
    if (no_show_ || num_top_persons_ <= 0) {
        return;
    }
    const int width = margin_size_ * (num_top_persons_ + 1) + crop_width_ * num_top_persons_;
    const int height = header_size_ + crop_height_ + margin_size_;
    top_persons_.create(height, width, CV_8UC3);
}

void DrawingHelper::ClearTopWindow() {
    if (no_show_ || num_top_persons_ <= 0) {
        return;
    }
    top_persons_.setTo(cv::Scalar(255, 255, 255));
    for (int i = 0; i < num_top_persons_; ++i) {
        const int shift = (i + 1) * margin_size_ + i * crop_width_;
        cv::rectangle(top_persons_,
                      cv::Point(shift, header_size_),
                      cv::Point(shift + crop_width_, header_size_ + crop_height_),
                      cv::Scalar(0, 0, 0),
                      cv::FILLED);

        const auto label_to_draw = "#" + std::to_string(i + 1);
        int baseLine = 0;
        const auto label_size = cv::getTextSize(label_to_draw, cv::FONT_HERSHEY_SIMPLEX, 2, 2, &baseLine);
        const int text_shift = (crop_width_ - label_size.width) / 2;
        cv::putText(top_persons_,
                    label_to_draw,
                    cv::Point(shift + text_shift, label_size.height + baseLine / 2),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1,
                    cv::Scalar(0, 255, 0),
                    2,
                    cv::LINE_AA);
    }
}

void DrawingHelper::ShowCrop(const cv::Mat& obj) {
    if (no_show_ || num_top_persons_ <= 0) {
        return;
    }
    if (!obj.empty()) {
        top_persons_ = top_persons_ + obj(cv::Rect(0, 0, top_persons_.cols, top_persons_.rows));
    }
    cv::imshow(top_window_name_, top_persons_);
}

void DrawingHelper::Finalize() {
    if (!no_show_) {
        cv::destroyWindow(main_window_name_);
        if (num_top_persons_ > 0) {
            cv::destroyWindow(top_window_name_);
        }
    }
}

void DrawingHelper::Show(const cv::Mat& frame) {
    if (!no_show_) {
        cv::imshow(main_window_name_, frame);
    }
}
