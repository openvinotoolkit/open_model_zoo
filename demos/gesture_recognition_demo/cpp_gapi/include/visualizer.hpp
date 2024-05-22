// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "tracker.hpp"

class Visualizer {
private:
    cv::Mat storageFrame;
    bool no_show_;
    std::string storage_window_name_;
    std::vector<std::string> labels_;
    std::string storage_path_;
    std::vector<std::pair<std::string, std::string>> storage_elements_;
    cv::VideoCapture gesture_cap_;
    int last_gesture = 0;
    std::string last_gesture_path = "";
    int last_action_ = -1;

    const cv::Scalar RED = {0, 0, 255};
    const cv::Scalar GREEN = {0, 255, 0};

    void getStorageElements();

    int getPlaceByKey(const int key);

    void updateCap(const std::string& path);

    void applyDrawing(const cv::Mat& frame,
                      const TrackedObjects out_detections,
                      const int out_label_number,
                      const size_t current_id);

public:
    Visualizer(const bool no_show,
               const std::string& storage_window_name,
               const std::vector<std::string>& labels,
               const std::string& storage_path)
        : no_show_(no_show),
          storage_window_name_(storage_window_name),
          labels_(labels),
          storage_path_(storage_path) {
        if (storage_path_.size() > 0) {
            getStorageElements();
        }
        if (no_show) {
            return;
        }

        if (storage_path_.size() > 0) {
            cv::namedWindow(storage_window_name_);
            gesture_cap_.open(storage_elements_.front().second);
        }
    }

    void show(const cv::Mat& frame,
              const TrackedObjects out_detections,
              const int out_label_number,
              const size_t current_id,
              const int key);
};
