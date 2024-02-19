// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include <opencv2/opencv.hpp>
#include <utils/slog.hpp>
#include "actions.hpp"
#include "tracker.hpp"

class DetectionsLogger {
private:
    bool m_write_logs;
    std::ofstream m_act_stat_log_stream;
    cv::FileStorage m_act_det_log_stream;
    slog::LogStream& m_log_stream;


public:
    explicit DetectionsLogger(slog::LogStream& stream, bool enabled,
                              const std::string& act_stat_log_file,
                              const std::string& act_det_log_file);

    ~DetectionsLogger();
    void CreateNextFrameRecord(const std::string& path, const int frame_idx,
                               const size_t width, const size_t height);
    void AddFaceToFrame(const cv::Rect& rect, const std::string& id, const std::string& action);
    void AddPersonToFrame(const cv::Rect& rect, const std::string& action, const std::string& id);
    void AddDetectionToFrame(const TrackedObject& object, const int frame_idx);
    void FinalizeFrameRecord();
    void DumpDetections(const std::string& video_path,
                        const cv::Size frame_size,
                        const size_t num_frames,
                        const std::vector<Track>& face_tracks,
                        const std::map<int, int>& track_id_to_label_faces,
                        const std::vector<std::string>& action_idx_to_label,
                        const std::vector<std::string>& person_id_to_label,
                        const std::vector<std::map<int, int>>& frame_face_obj_id_to_action_maps);
    void DumpTracks(const std::map<int, RangeEventsTrack>& obj_id_to_events,
                    const std::vector<std::string>& action_idx_to_label,
                    const std::map<int, int>& track_id_to_label_faces,
                    const std::vector<std::string>& person_id_to_label);
};


#define SCR_CHECK(cond) CV_Assert(cond);

#define SCR_CHECK_BINARY(actual, expected, op) \
    CV_Assert(actual op expected);

#define SCR_CHECK_EQ(actual, expected) SCR_CHECK_BINARY(actual, expected, ==)
#define SCR_CHECK_NE(actual, expected) SCR_CHECK_BINARY(actual, expected, !=)
#define SCR_CHECK_LT(actual, expected) SCR_CHECK_BINARY(actual, expected, <)
#define SCR_CHECK_GT(actual, expected) SCR_CHECK_BINARY(actual, expected, >)
#define SCR_CHECK_LE(actual, expected) SCR_CHECK_BINARY(actual, expected, <=)
#define SCR_CHECK_GE(actual, expected) SCR_CHECK_BINARY(actual, expected, >=)
