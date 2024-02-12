// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "logger.hpp"

#include <algorithm>
#include <iomanip>
#include <memory>
#include <utility>

#include "tracker.hpp"

namespace {

const char unknown_label[] = "Unknown";

std::string GetUnknownOrLabel(const std::vector<std::string>& labels, int idx) {
    return idx >= 0 ? labels.at(idx) : unknown_label;
}

std::string FrameIdxToString(const std::string& path, int frame_idx) {
    std::stringstream ss;
    ss << std::setw(6) << std::setfill('0') << frame_idx;
    return path.substr(path.rfind("/") + 1) + "@" + ss.str();
}
}  // anonymous namespace

DetectionsLogger::DetectionsLogger() {}

DetectionsLogger::DetectionsLogger(bool enabled) : write_logs_(enabled) {}

void DetectionsLogger::CreateNextFrameRecord(const std::string& path,
                                             const int frame_idx,
                                             const size_t width,
                                             const size_t height) {
    if (write_logs_) {
        log_stream_.clear();
        log_stream_ << "Frame_name: " << path << "@" << frame_idx << " width: " << width << " height: " << height
                    << std::endl;
    }
}

void DetectionsLogger::AddFaceToFrame(const cv::Rect& rect, const std::string& id, const std::string& action) {
    if (write_logs_) {
        log_stream_ << "Object type: face. Box: " << rect << " id: " << id;
        if (!action.empty()) {
            log_stream_ << " action: " << action;
        }
        log_stream_ << std::endl;
    }
}

void DetectionsLogger::AddPersonToFrame(const cv::Rect& rect, const std::string& action, const std::string& id) {
    if (write_logs_) {
        log_stream_ << "Object type: person. Box: " << rect << " action: " << action;
        if (!id.empty()) {
            log_stream_ << " id: " << id;
        }
        log_stream_ << std::endl;
    }
}

void DetectionsLogger::AddDetectionToFrame(const TrackedObject& object, const int frame_idx) {
    act_det_log_stream_ << "{"
                        << "frame_id " << frame_idx;
    act_det_log_stream_ << " det_conf " << object.confidence << std::endl;
    act_det_log_stream_ << "label " << object.label << std::endl;
    act_det_log_stream_ << "rect " << object.rect << "}" << std::endl;
}

void DetectionsLogger::FinalizeFrameRecord() {
    if (write_logs_) {
        log_stream_ << std::endl;
    }
}

void DetectionsLogger::DumpDetections(const std::string& video_path,
                                      const cv::Size frame_size,
                                      const size_t num_frames,
                                      const std::vector<Track>& face_tracks,
                                      const std::map<int, int>& track_id_to_label_faces,
                                      const std::vector<std::string>& action_idx_to_label,
                                      const std::vector<std::string>& person_id_to_label,
                                      const std::vector<std::map<int, int>>& frame_face_obj_id_to_action_maps) {
    std::map<int, std::vector<const TrackedObject*>> frame_idx_to_face_track_objs;

    for (const auto& tr : face_tracks) {
        for (const auto& obj : tr.objects) {
            frame_idx_to_face_track_objs[obj.frame_idx].emplace_back(&obj);
        }
    }

    std::map<std::string, std::string> face_label_to_action;
    for (const auto& label : person_id_to_label) {
        face_label_to_action[label] = unknown_label;
    }
    act_stat_log_stream_ << "frame_idx";
    for (const auto& label : person_id_to_label) {
        act_stat_log_stream_ << "," << label;
    }
    act_stat_log_stream_ << std::endl;

    for (size_t i = 0; i < num_frames; i++) {
        CreateNextFrameRecord(video_path, i, frame_size.width, frame_size.height);
        const auto& frame_face_obj_id_to_action = frame_face_obj_id_to_action_maps.at(i);
        for (auto& kv : face_label_to_action) {
            kv.second = unknown_label;
        }

        for (const auto& p_obj : frame_idx_to_face_track_objs[i]) {
            const auto& obj = *p_obj;
            std::string action_label = unknown_label;
            if (frame_face_obj_id_to_action.count(obj.object_id) > 0) {
                action_label = GetUnknownOrLabel(action_idx_to_label, frame_face_obj_id_to_action.at(obj.object_id));
            }
            std::string face_label = GetUnknownOrLabel(person_id_to_label, track_id_to_label_faces.at(obj.object_id));
            face_label_to_action[face_label] = action_label;
            AddFaceToFrame(obj.rect, face_label, action_label);
        }

        act_stat_log_stream_ << FrameIdxToString(video_path, i);
        for (const auto& label : person_id_to_label) {
            act_stat_log_stream_ << "," << face_label_to_action[label];
        }
        act_stat_log_stream_ << std::endl;
        FinalizeFrameRecord();
    }
}

void DetectionsLogger::DumpTracks(const std::map<int, RangeEventsTrack>& obj_id_to_events,
                                  const std::vector<std::string>& action_idx_to_label,
                                  const std::map<int, int>& track_id_to_label_faces,
                                  const std::vector<std::string>& person_id_to_label) {
    for (const auto& tup : obj_id_to_events) {
        const int obj_id = tup.first;
        if (track_id_to_label_faces.count(obj_id) > 0) {
            const auto& events = tup.second;
            std::string face_label = GetUnknownOrLabel(person_id_to_label, track_id_to_label_faces.at(obj_id));
            log_stream_ << "Person: " << face_label << std::endl;
            for (const auto& event : events) {
                std::string action_label = GetUnknownOrLabel(action_idx_to_label, event.action);
                log_stream_ << "   - " << action_label << ": from " << event.begin_frame_id << " to "
                            << event.end_frame_id << " frames" << std::endl;
            }
        }
    }
}

std::tuple<std::string, std::string, std::string> DetectionsLogger::GetLogResult() {
    return std::make_tuple(log_stream_.str(), act_stat_log_stream_.str(), act_det_log_stream_.str());
}

void DetectionsLogger::ConvertActionMapsToFrameEventTracks(const std::vector<std::map<int, int>>& obj_id_to_action_maps,
                                                           int default_action,
                                                           std::map<int, FrameEventsTrack>* obj_id_to_actions_track) {
    for (size_t frame_id = 0; frame_id < obj_id_to_action_maps.size(); ++frame_id) {
        for (const auto& tup : obj_id_to_action_maps[frame_id]) {
            if (tup.second != default_action) {
                (*obj_id_to_actions_track)[tup.first].emplace_back(frame_id, tup.second);
            }
        }
    }
}

void DetectionsLogger::ConvertRangeEventsTracksToActionMaps(int num_frames,
                                                            const std::map<int, RangeEventsTrack>& obj_id_to_events,
                                                            std::vector<std::map<int, int>>* obj_id_to_action_maps) {
    obj_id_to_action_maps->resize(num_frames);
    for (const auto& tup : obj_id_to_events) {
        const int obj_id = tup.first;
        const auto& events = tup.second;
        for (const auto& event : events) {
            for (int frame_id = event.begin_frame_id; frame_id < event.end_frame_id; ++frame_id) {
                (*obj_id_to_action_maps)[frame_id].emplace(obj_id, event.action);
            }
        }
    }
}

void DetectionsLogger::SmoothTracks(const std::map<int, FrameEventsTrack>& obj_id_to_actions_track,
                                    int start_frame,
                                    int end_frame,
                                    int window_size,
                                    int min_length,
                                    int default_action,
                                    std::map<int, RangeEventsTrack>* obj_id_to_events) {
    // Iterate over face tracks
    for (const auto& tup : obj_id_to_actions_track) {
        const auto& frame_events = tup.second;
        if (frame_events.empty()) {
            continue;
        }
        RangeEventsTrack range_events;

        // Merge neighbouring events and filter short ones
        range_events.emplace_back(frame_events.front().frame_id,
                                  frame_events.front().frame_id + 1,
                                  frame_events.front().action);

        for (size_t frame_id = 1; frame_id < frame_events.size(); ++frame_id) {
            const auto& last_range_event = range_events.back();
            const auto& cur_frame_event = frame_events[frame_id];

            if (last_range_event.end_frame_id + window_size - 1 >= cur_frame_event.frame_id &&
                last_range_event.action == cur_frame_event.action) {
                range_events.back().end_frame_id = cur_frame_event.frame_id + 1;
            } else {
                if (range_events.back().end_frame_id - range_events.back().begin_frame_id < min_length) {
                    range_events.pop_back();
                }

                range_events.emplace_back(cur_frame_event.frame_id,
                                          cur_frame_event.frame_id + 1,
                                          cur_frame_event.action);
            }
        }
        if (range_events.back().end_frame_id - range_events.back().begin_frame_id < min_length) {
            range_events.pop_back();
        }

        // Extrapolate track
        if (range_events.empty()) {
            range_events.emplace_back(start_frame, end_frame, default_action);
        } else {
            range_events.front().begin_frame_id = start_frame;
            range_events.back().end_frame_id = end_frame;
        }

        // Interpolate track
        for (size_t event_id = 1; event_id < range_events.size(); ++event_id) {
            auto& last_event = range_events[event_id - 1];
            auto& cur_event = range_events[event_id];

            int middle_point = static_cast<int>(0.5f * (cur_event.begin_frame_id + last_event.end_frame_id));

            cur_event.begin_frame_id = middle_point;
            last_event.end_frame_id = middle_point;
        }

        // Merge consecutive events
        auto& final_events = (*obj_id_to_events)[tup.first];
        final_events.push_back(range_events.front());
        for (size_t event_id = 1; event_id < range_events.size(); ++event_id) {
            const auto& cur_event = range_events[event_id];

            if (final_events.back().action == cur_event.action) {
                final_events.back().end_frame_id = cur_event.end_frame_id;
            } else {
                final_events.push_back(cur_event);
            }
        }
    }
}

std::map<int, int> DetectionsLogger::GetMapFaceTrackIdToLabel(const std::vector<Track>& face_tracks) {
    std::map<int, int> face_track_id_to_label;
    for (const auto& track : face_tracks) {
        const auto& first_obj = track.first_object;
        // check consistency
        // to receive this consistency for labels
        // use the function UpdateTrackLabelsToBestAndFilterOutUnknowns
        for (const auto& obj : track.objects) {
            SCR_CHECK_EQ(obj.label, first_obj.label);
            SCR_CHECK_EQ(obj.object_id, first_obj.object_id);
        }

        auto cur_obj_id = first_obj.object_id;
        auto cur_label = first_obj.label;
        if (face_track_id_to_label.count(cur_obj_id) == 1) {
            throw std::runtime_error("Repeating face tracks");
        }
        face_track_id_to_label[cur_obj_id] = cur_label;
    }
    return face_track_id_to_label;
}
