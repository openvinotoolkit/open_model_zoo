// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

struct TrackedObject;

class DrawingHelper {
public:
    static const int crop_width_ = 128;
    static const int crop_height_ = 320;
    static const int header_size_ = 80;
    static const int margin_size_ = 5;
    static const int max_input_width_ = 1920;
    float rect_scale_x_ = 0;
    float rect_scale_y_ = 0;
    const std::string main_window_name_ = "Smart classroom demo G-API";
    const std::string top_window_name_ = "Top-k students";
    bool no_show_ = false;
    int num_top_persons_ = -1;
    cv::Mat top_persons_;

    DrawingHelper(bool show, int num_top_persons);
    static cv::Size GetOutputSize(const cv::Size& input_size);
    float CalculateIoM(const cv::Rect& rect1, const cv::Rect& rect2);
    cv::Rect DecreaseRectByRelBorders(const cv::Rect& r);
    int GetIndexOfTheNearestPerson(const TrackedObject& face, const std::vector<TrackedObject>& tracked_persons);
    std::string GetActionTextLabel(const unsigned label, const std::vector<std::string>& actions_map);
    void GetNewFrameSize(const cv::Size& frame_size);
    void CreateTopWindow();
    void ClearTopWindow();
    void Finalize();
    void Show(const cv::Mat& frame);
    void ShowCrop(const cv::Mat& obj = cv::Mat());
};

struct DrawingElements {
    std::vector<cv::Rect> rects_det;
    std::vector<cv::Rect> rects_face;
    std::vector<std::string> labels_det;
    std::vector<std::string> labels_face;
};
