// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <opencv2/opencv.hpp>
#include <details/ie_exception.hpp>

#include "tracker.hpp"
#include "actions.hpp"

class DetectionsLogger {
private:
    bool write_logs_ = false;

    std::stringstream act_stat_log_stream_;
    std::stringstream act_det_log_stream_;
    std::stringstream log_stream_;
    const int default_action_index = -1; // Unknown action class
public:
    DetectionsLogger();
    DetectionsLogger(bool enabled);

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
    std::tuple<std::string, std::string, std::string> GetLogResult();
    void ConvertActionMapsToFrameEventTracks(const std::vector<std::map<int, int>>& obj_id_to_action_maps,
                                             int default_action, std::map<int,
                                             FrameEventsTrack>* obj_id_to_actions_track);
    void ConvertRangeEventsTracksToActionMaps(int num_frames, const std::map<int,
                                              RangeEventsTrack>& obj_id_to_events,
                                              std::vector<std::map<int, int>>* obj_id_to_action_maps);
    void SmoothTracks(const std::map<int, FrameEventsTrack>& obj_id_to_actions_track,
                      int start_frame, int end_frame, int window_size, int min_length,
                      int default_action, std::map<int, RangeEventsTrack>* obj_id_to_events);
    std::map<int, int> GetMapFaceTrackIdToLabel(const std::vector<Track>& face_tracks);
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
