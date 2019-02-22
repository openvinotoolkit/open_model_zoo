// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <map>
#include <set>
#include <vector>
#include <fstream>

#include "logger.hpp"

namespace {

const std::string unknown_label = "Unknown";

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
                                   const std::string& act_log_file) : log_stream_(stream) {
    write_logs_ = enabled;
    act_log_stream_.open(act_log_file, std::fstream::out);
}

void DetectionsLogger::CreateNextFrameRecord(const std::string& path, const int frame_idx,
                                             const size_t width, const size_t height) {
    if (write_logs_)
        log_stream_ << "Frame_name: " << path << "@" << frame_idx << " width: "
                    << width << " height: " << height << std::endl;
}

void DetectionsLogger::AddFaceToFrame(const cv::Rect& rect, const std::string& id, const std::string& action) {
    if (write_logs_)
        log_stream_ << "Object type: face. Box: " << rect << " id: " << id << " action: " << action << std::endl;
}

void DetectionsLogger::AddPersonToFrame(const cv::Rect& rect, const std::string& action, const std::string& id) {
    if (write_logs_)
        log_stream_ << "Object type: person. Box: " << rect << " action: " << action << " id: " << id << std::endl;
}

void DetectionsLogger::FinalizeFrameRecord() {
    if (write_logs_)
        log_stream_ << std::endl;
}

void DetectionsLogger::DumpDetections(const std::string& video_path,
                                      const cv::Size frame_size,
                                      const int num_frames,
                                      const std::vector<Track>& face_tracks,
                                      const std::map<int, int>& track_id_to_label_faces,
                                      const std::vector<std::string>& action_idx_to_label,
                                      const std::vector<std::string>& person_id_to_label,
                                      const std::vector<std::map<int, int>>& frame_face_obj_id_to_action_maps)  {
    std::map<int, std::vector<const TrackedObject*>> frame_idx_to_face_track_objs;

    for (const auto& tr : face_tracks) {
        int cur_tr_id = tr.first_object.object_id;
        for (const auto& obj : tr.objects) {
            frame_idx_to_face_track_objs[obj.frame_idx].emplace_back(&obj);
        }
    }

    std::map<std::string, std::string> face_label_to_action;
    for (const auto& label : person_id_to_label) {
        face_label_to_action[label] = unknown_label;
    }
    act_log_stream_ << "frame_idx";
    for (const auto& label : person_id_to_label) {
        act_log_stream_ << "," << label;
    }
    act_log_stream_ << std::endl;

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

        act_log_stream_ << FrameIdxToString(video_path, i);
        for (const auto& label : person_id_to_label) {
            act_log_stream_ << "," << face_label_to_action[label];
        }
        act_log_stream_ << std::endl;

        FinalizeFrameRecord();
    }
}

DetectionsLogger::~DetectionsLogger() {}
