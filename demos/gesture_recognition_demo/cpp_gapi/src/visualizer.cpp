// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define _USE_MATH_DEFINES

#include "visualizer.hpp"

#include <algorithm>

#include <opencv2/imgproc.hpp>

void Visualizer::getStorageElements() {
    cv::FileStorage fs(storage_path_, cv::FileStorage::Mode::READ);
    cv::FileNode fn = fs.root();

    for (auto fit = fn.begin(); fit != fn.end(); ++fit) {
        cv::FileNode item = *fit;
        std::string label = item.name();
        storage_elements_.emplace_back(item.name(), item[0].string());
    }
}

int Visualizer::getPlaceByKey(const int key) {
    const auto gestures_size = static_cast<int>(storage_elements_.size());
    int new_gesture = last_gesture + key;
    if (new_gesture < 0) {
        new_gesture *= -1;
    }
    if (new_gesture > gestures_size) {
        new_gesture -= gestures_size;
    }
    last_gesture = new_gesture;
    return new_gesture;
}

void Visualizer::updateCap(const std::string& path) {
    if (gesture_cap_.isOpened()) {
        gesture_cap_.release();
    }
    gesture_cap_.open(path);
}

void Visualizer::applyDrawing(const cv::Mat& frame,
                              const TrackedObjects out_detections,
                              const int out_label_number,
                              const size_t current_id) {
    cv::Scalar color;
    for (const auto& person_id_roi : out_detections) {
        const size_t id = size_t(person_id_roi.object_id);
        const cv::Rect bb = person_id_roi.rect;

        color = id == current_id ? GREEN : RED;
        cv::putText(frame, std::to_string(id), cv::Point(bb.x + 10, bb.y + 30), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
        cv::rectangle(frame, bb, color, 2);
    }

    if (out_label_number >= 0) {
        last_action_ = out_label_number;
    }
    std::string label = std::string("Last gesture: ") + std::string(last_action_ > 0 ? labels_[last_action_] : "");
    cv::putText(frame, label, cv::Point(40, frame.rows - 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, RED, 2);
}

void Visualizer::show(const cv::Mat& frame,
                      const TrackedObjects out_detections,
                      const int out_label_number,
                      const size_t current_id,
                      const int key) {
    applyDrawing(frame, out_detections, out_label_number, current_id);
    if (!no_show_) {
        cv::imshow("Gesture Recognition demo G-API", frame);
        if (!storage_path_.empty()) {
            const auto place = getPlaceByKey(key);
            const auto new_path = storage_elements_.at(place).second;
            if (last_gesture_path != new_path) {
                updateCap(new_path);
                last_gesture_path = new_path;
            }
            cv::Mat gesture_mat;
            gesture_cap_.read(gesture_mat);

            cv::putText(gesture_mat,
                        storage_elements_.at(place).first,
                        cv::Point(20, 20),
                        cv::FONT_HERSHEY_SIMPLEX,
                        2,
                        RED);
            cv::imshow(storage_window_name_, gesture_mat);
        }
    }
}
