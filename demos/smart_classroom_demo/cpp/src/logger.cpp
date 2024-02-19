// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iomanip>
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

DetectionsLogger::DetectionsLogger(slog::LogStream& stream, bool enabled,
                                   const std::string& act_stat_log_file,
                                   const std::string& act_det_log_file) :
    m_log_stream(stream) {
    m_write_logs = enabled;
    m_act_stat_log_stream.open(act_stat_log_file, std::fstream::out);

    if (!act_det_log_file.empty()) {
        m_act_det_log_stream.open(act_det_log_file, cv::FileStorage::WRITE);

        m_act_det_log_stream << "data" << "[";
    }
}

void DetectionsLogger::CreateNextFrameRecord(
    const std::string& path, const int frame_idx, const size_t width, const size_t height) {
    if (m_write_logs)
        m_log_stream << "Frame_name: " << path << "@" << frame_idx << " width: "
                     << width << " height: " << height << slog::endl;
}

void DetectionsLogger::AddFaceToFrame(const cv::Rect& rect, const std::string& id, const std::string& action) {
    if (m_write_logs) {
        m_log_stream << "Object type: face. Box: " << rect << " id: " << id;
        if (!action.empty()) {
            m_log_stream << " action: " << action;
        }
        m_log_stream << slog::endl;
    }
}

void DetectionsLogger::AddPersonToFrame(const cv::Rect& rect, const std::string& action, const std::string& id) {
    if (m_write_logs) {
        m_log_stream << "Object type: person. Box: " << rect << " action: " << action;
        if (!id.empty()) {
            m_log_stream << " id: " << id;
        }
        m_log_stream << slog::endl;
    }
}

void DetectionsLogger::AddDetectionToFrame(const TrackedObject& object, const int frame_idx) {
    if (m_act_det_log_stream.isOpened()) {
        m_act_det_log_stream << "{" << "frame_id" << frame_idx
                             << "det_conf" << object.confidence
                             << "label" << object.label
                             << "rect" << object.rect << "}";
    }
}

void DetectionsLogger::FinalizeFrameRecord() {
    if (m_write_logs) {
        m_log_stream << slog::endl;
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
    m_act_stat_log_stream << "frame_idx";
    for (const auto& label : person_id_to_label) {
        m_act_stat_log_stream << "," << label;
    }
    m_act_stat_log_stream << std::endl;

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

        m_act_stat_log_stream << FrameIdxToString(video_path, i);
        for (const auto& label : person_id_to_label) {
            m_act_stat_log_stream << "," << face_label_to_action[label];
        }
        m_act_stat_log_stream << std::endl;

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
            m_log_stream << "Person: " << face_label << slog::endl;

            for (const auto& event : events) {
                std::string action_label = GetUnknownOrLabel(action_idx_to_label, event.action);
                m_log_stream << "   - " << action_label
                             << ": from " << event.begin_frame_id
                             << " to " << event.end_frame_id
                             << " frames" << slog::endl;
            }
        }
    }
}

DetectionsLogger::~DetectionsLogger() {
    if (m_act_det_log_stream.isOpened()) {
        m_act_det_log_stream << "]";
    }
}
