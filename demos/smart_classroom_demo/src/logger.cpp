// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <map>
#include <set>
#include <vector>
#include <fstream>

#include "logger.hpp"

namespace {

const char unknown_label[] = "Unknown";

std::string GetUnknownOrLabel(const std::vector<std::string>& labels, int idx)  {
    return idx >= 0 ? labels.at(idx) : unknown_label;
}

std::string FrameIdxToString(const std::string& path, int frame_idx) {
    std::stringstream ss;
    ss << std::setw(6) << std::setfill('0') << frame_idx;
    return path.substr(path.rfind("/") + 1) + "@" + ss.str();
}
}  // anonymous namespace

DetectionsLogger::DetectionsLogger(std::ostream& stream, bool enabled,
                                   const std::string& act_stat_log_file,
                                   const std::string& act_det_log_file)
    : log_stream_(stream) {
    write_logs_ = enabled;
    act_stat_log_stream_.open(act_stat_log_file, std::fstream::out);

    if (!act_det_log_file.empty()) {
        act_det_log_stream_.open(act_det_log_file, cv::FileStorage::WRITE);

        act_det_log_stream_ << "data" << "[";
    }
}

void DetectionsLogger::CreateNextFrameRecord(const std::string& path, const int frame_idx,
                                             const size_t width, const size_t height) {
    if (write_logs_)
        log_stream_ << "Frame_name: " << path << "@" << frame_idx << " width: "
                    << width << " height: " << height << std::endl;
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
    if (act_det_log_stream_.isOpened()) {
        act_det_log_stream_ << "{" << "frame_id" << frame_idx
                            << "det_conf" << object.confidence
                            << "label" << object.label
                            << "rect" << object.rect << "}";
    }
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
                                      const std::vector<std::map<int, int>>& frame_face_obj_id_to_action_maps)  {
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

    for (size_t i = 0; i < num_frames; i++)  {
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
                log_stream_ << "   - " << action_label
                            << ": from " << event.begin_frame_id
                            << " to " << event.end_frame_id
                            << " frames" <<std::endl;
            }
        }
    }
}

DetectionsLogger::~DetectionsLogger() {
    if (act_det_log_stream_.isOpened()) {
        act_det_log_stream_ << "]";
    }
}
