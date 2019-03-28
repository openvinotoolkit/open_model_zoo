// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <deque>
#include <set>
#include <unordered_map>
#include <map>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <details/ie_exception.hpp>
#include "tracker.hpp"

#include "actions.hpp"

class DetectionsLogger {
private:
    bool write_logs_;
    std::ofstream act_stat_log_stream_;
    cv::FileStorage act_det_log_stream_;
    std::ostream& log_stream_;

public:
    explicit DetectionsLogger(std::ostream& stream, bool enabled,
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


#define SCR_CHECK(cond) IE_ASSERT(cond) << " "

#define SCR_CHECK_BINARY(actual, expected, op) \
    IE_ASSERT(actual op expected) << ". " \
        << actual << " vs " << expected << ".  "

#define SCR_CHECK_EQ(actual, expected) SCR_CHECK_BINARY(actual, expected, ==)
#define SCR_CHECK_NE(actual, expected) SCR_CHECK_BINARY(actual, expected, !=)
#define SCR_CHECK_LT(actual, expected) SCR_CHECK_BINARY(actual, expected, <)
#define SCR_CHECK_GT(actual, expected) SCR_CHECK_BINARY(actual, expected, >)
#define SCR_CHECK_LE(actual, expected) SCR_CHECK_BINARY(actual, expected, <=)
#define SCR_CHECK_GE(actual, expected) SCR_CHECK_BINARY(actual, expected, >=)
